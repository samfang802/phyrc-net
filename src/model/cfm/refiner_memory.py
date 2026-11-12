"""Monotone structure memory for the refiner (ILF, docs/redesign/phase1_ilf.md).

ILF-lite (fixed rule): thin dark structure of a frame = ``min(0, gray - blur_sigma(gray) + tau)`` (cracks, grid
lines, spall edges darker than ``tau`` below their surroundings). A pixel is written into the memory once it is
dark in most of the last ``window`` frames (temporal median), and the memory is the running minimum of those
writes, so established structure stays while one-frame changes (crack opening, exposure, small shifts) do not
accumulate. On the real sequences this keeps the end-of-walk memory at ~2.5x a single frame's density
(a plain running min of single frames reaches ~80% of the image by the last frame).

Full ILF switches, each usable alone:
  Kaiser gate (``refiner_memory_gate=kaiser``): the memory is written only on frames of the first cycle at a
    new drift level (filename suffix ``_0`` / ``_1`` / ``_2``), the ones that reach a new load envelope.
  Learned encoder (``refiner_memory_encoder``): ``m_t = max(m_{t-1}, gate_t * E(I_t))``, E a small conv net to a
    non-negative ``C_m``-channel field at 1/4 resolution, trained end to end through the max.
  Frozen trunk: the existing ``--freeze_trunk``.
  Encoder input (``refiner_memory_input``, ``memory_input``): raw gray; ``hp`` gray - blur_2(gray); ``delta``
    gray(I_t) - gray(I_0) minus its per-image median; ``hpnew`` thin dark structure of I_t not within 2 px of
    thin dark structure of I_0, ``min(0, min(0, hp_t) - minpool5(min(0, hp_0)))`` (camera / grid jitter). I_0 = the walk's first frame
    (real at training, the generated frame 0 at sampling); delta / hpnew cancel the static grid / background / hoop.
  Decoder auxiliary loss (``refiner_memory_aux_weight``): a small decoder D reads the memory field and predicts the
    accumulated darkening of the encoder input, ``max over the earlier writing frames of relu(-input - 0.03)`` at full
    resolution (0.03 keeps the jitter floor out of the running max), so the 1/4-resolution field has to hold where
    the thin structure is. ``refiner_memory_aux_loss``: ``l1`` normalised L1 on that depth (collapses to 0 on the
    sparse target: the per-pixel median), ``bce_dice`` BCE (batch pos_weight) + Dice on its binary version (> 0).
  Encoder input scale (``refiner_memory_input_scale``): E reads ``scale * input`` (hpnew is ~0 to -0.3; 20 brings it
    to O(1)). Contrast augmentation (``refiner_memory_contrast_aug`` = a < 1, training only): each written frame's
    input is multiplied by alpha ~ U[a, 1] (generated frames are fogged, i.e. contrast-compressed: their hpnew is
    ~40-60% of the real one); the decoder target stays unaugmented. Sparsity (``refiner_memory_sparse_weight``):
    L1 on E's output so cells with nothing new stay 0, the value the memory drop / guidance use for "no memory".

Self-confirmation switches (rule memory, docs/redesign/phase1_ilf.md §8):
  Novelty gate (``refiner_memory_gate=novelty``): a write is committed only if it adds at least
    ``refiner_memory_novelty`` of the image as new entries.
  TTL (``refiner_memory_ttl``): an entry not visible (within 1 px) in any of the last ``ttl`` frames is cleared.
  Error recycling (``refiner_memory_recycle``, training only): the real memory is corrupted with the model's own
    structure errors (``memory_error`` of x̂1 against the real frame, banked per frame row during training), from
    earlier frames of the same walk or transplanted from other frames, so the refiner learns not to redraw
    memory entries the conditions do not support.

The refiner reads the memory before frame t as extra control-hint channels next to ``S_{t-1}``. Training reads
the real frames' memory (built without augmentation, flipped with the sample); sampling accumulates the
generated frames' memory with the same rule.
"""

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SUFFIX = re.compile(r"_(\d+)\.[A-Za-z]+$")


def kaiser_gate(filename):
    """True on the first cycle of a drift level (frame suffix 0-2 of the 6 frames = 3 cycles x 2 signs)."""
    m = _SUFFIX.search(str(filename))
    if m is None:
        raise ValueError(f"no frame suffix in {filename!r}")
    return int(m.group(1)) <= 2


def write_gates(filenames, gate_mode):
    """Per-frame write flags from the filenames (the novelty gate is decided inside ``MemoryTrack``)."""
    if gate_mode in ("none", "novelty"):
        return [True] * len(filenames)
    if gate_mode == "kaiser":
        return [kaiser_gate(f) for f in filenames]
    raise ValueError(f"refiner_memory_gate must be none|kaiser|novelty, got {gate_mode!r}")


def _gauss_1d(sigma, device, dtype):
    r = max(1, int(math.ceil(3.0 * float(sigma))))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum(), r


def memory_feature(x, sigma=2.0, tau=0.15):
    """x [B, C, H, W] model units -> [B, 1, H, W] <= 0."""
    g = x.float().mean(dim=1, keepdim=True)
    k, r = _gauss_1d(sigma, g.device, g.dtype)
    blur = F.conv2d(F.pad(g, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    blur = F.conv2d(F.pad(blur, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return (g - blur + float(tau)).clamp(max=0.0).to(x.dtype)


def memory_error(x_hat, x, sigma=2.0, tau=0.15):
    """[B, 1, H, W] structure error of a model frame against the real one: < 0 where the model draws structure
    the real frame lacks, > 0 where it misses real structure."""
    return memory_feature(x_hat, sigma, tau).float() - memory_feature(x, sigma, tau).float()


def inject_memory_errors(mem, errs, valid):
    """mem [B, 1, H, W] <= 0, errs [B, n, H, W], valid [B, n] bool. The spurious structure of every error is added
    (a union, as the running min keeps it); real structure is removed only where every error misses it."""
    v = valid.view(*valid.shape, 1, 1)
    spur = torch.where(v, errs.clamp(max=0.0), torch.zeros_like(errs)).amin(dim=1, keepdim=True)
    miss = torch.where(v, errs.clamp(min=0.0), torch.full_like(errs, float("inf"))).amin(dim=1, keepdim=True)
    miss = torch.where(valid.any(dim=1).view(-1, 1, 1, 1), miss, torch.zeros_like(miss))
    return torch.minimum((mem + miss).clamp(max=0.0), spur).to(mem.dtype)


class MemoryTrack:
    """Rule memory of one walk (any batch of trajectories side by side). ``novelty`` > 0: a write is committed
    per trajectory only if it adds at least that share of the image as new entries. ``ttl`` > 0: an entry that is
    not visible (within 1 px) in the frames for more than ``ttl`` updates is cleared. Entries: memory < -eps."""

    def __init__(self, sigma=2.0, tau=0.15, window=7, ttl=0, novelty=0.0, eps=0.03):
        self.sigma, self.tau, self.window = float(sigma), float(tau), int(window)
        self.ttl, self.novelty, self.eps = int(ttl), float(novelty), float(eps)
        self.recent = []
        self.value = None
        self.last = None
        self.n = 0

    def update(self, frame, gate=True):
        feat = memory_feature(frame, self.sigma, self.tau)
        self.recent = (self.recent + [feat])[-self.window:]
        if self.value is None:
            self.value = torch.zeros_like(feat)
            self.last = torch.zeros_like(feat, dtype=torch.float32)
        held_before = self.value < -self.eps
        if gate:
            write = torch.stack(self.recent).median(dim=0).values
            if self.novelty > 0.0:
                new = (write < -self.eps) & ~held_before
                ok = new.flatten(1).float().mean(dim=1) >= self.novelty
                write = torch.where(ok.view(-1, 1, 1, 1), write, torch.zeros_like(write))
            self.value = torch.minimum(self.value, write)
        if self.ttl > 0:
            held = self.value < -self.eps
            seen = F.max_pool2d((feat < -self.eps).float(), 3, stride=1, padding=1) > 0
            self.last = torch.where((held & seen) | (held & ~held_before), torch.full_like(self.last, self.n), self.last)
            stale = held & (self.n - self.last > self.ttl)
            self.value = torch.where(stale, torch.zeros_like(self.value), self.value)
        self.n += 1
        return self.value


MEMORY_INPUTS = ("raw", "hp", "delta", "hpnew")


def _highpass(g, sigma=2.0):
    k, r = _gauss_1d(sigma, g.device, g.dtype)
    blur = F.conv2d(F.pad(g, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    blur = F.conv2d(F.pad(blur, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return g - blur


def memory_input(frame, ref, mode):
    """[B, C, H, W] frame (and the walk-start frame ``ref``, same shape or [1, ...]) -> [B, 1, H, W] encoder input."""
    g = frame.float().mean(dim=1, keepdim=True)
    if mode == "raw":
        return g
    if mode == "hp":
        return _highpass(g)
    g0 = ref.float().mean(dim=1, keepdim=True).expand_as(g)
    if mode == "delta":
        d = g - g0
        return d - d.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    if mode == "hpnew":
        f, f0 = _highpass(g).clamp(max=0.0), _highpass(g0).clamp(max=0.0)
        return (f + F.max_pool2d(-f0, 5, stride=1, padding=2)).clamp(max=0.0)
    raise ValueError(f"refiner_memory_input must be one of {MEMORY_INPUTS}, got {mode!r}")


class MemoryEncoder(nn.Module):
    """E: encoder input [B, C, H, W] (``memory_input``; the training bank already holds it) -> non-negative
    [B, C_m, H/4, W/4]. ``aux``: also a decoder D, field -> [B, 1, H, W] accumulated darkening."""

    def __init__(self, channels=4, width=32, input_mode="raw", aux=False, input_scale=1.0, aux_loss="l1"):
        super().__init__()
        if input_mode not in MEMORY_INPUTS:
            raise ValueError(f"refiner_memory_input must be one of {MEMORY_INPUTS}, got {input_mode!r}")
        if aux_loss not in ("l1", "bce_dice"):
            raise ValueError(f"refiner_memory_aux_loss must be l1|bce_dice, got {aux_loss!r}")
        self.input_mode = str(input_mode)
        self.input_scale = float(input_scale)
        self.aux_loss = str(aux_loss)
        self.net = nn.Sequential(
            nn.Conv2d(1, width // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, int(channels), 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.fill_(-2.0)  # softplus(-2) ~ 0.13: starts nearly empty
        self.decoder = None
        if aux:
            self.decoder = nn.Sequential(
                nn.Conv2d(int(channels), width, 3, padding=1), nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(width, width // 2, 3, padding=1), nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(width // 2, 1, 3, padding=1),
            )
            with torch.no_grad():
                self.decoder[-1].weight.mul_(0.1)
                # l1: softplus(-5) ~ 0.007, predicts no darkening (loss ~1); bce_dice: logit -2 ~ 12% positive
                self.decoder[-1].bias.fill_(-5.0 if aux_loss == "l1" else -2.0)

    def forward(self, x):
        return F.softplus(self.net(x.mean(dim=1, keepdim=True) * getattr(self, "input_scale", 1.0)))

    def decode(self, m):
        """Memory field [B, C_m, h, w] -> [B, 1, 4h, 4w] predicted accumulated darkening (>= 0) for ``l1``, the
        probability of accumulated structure for ``bce_dice``."""
        out = self.decoder(m)
        return torch.sigmoid(out) if getattr(self, "aux_loss", "l1") == "bce_dice" else F.softplus(out)

    def aux_loss_value(self, m, target):
        """Decoder loss of the fields m [B, C_m, h, w] against ``aux_target`` maps [B, 1, H, W]. Returns
        (loss, dict of float parts)."""
        out = self.decoder(m)
        if getattr(self, "aux_loss", "l1") == "l1":
            loss = (F.softplus(out) - target).abs().sum() / target.sum().clamp_min(1.0)
            return loss, {"l1": float(loss.detach())}
        y = (target > 0).float()
        pos = y.mean().clamp(1e-4, 1.0)
        pw = ((1.0 - pos) / pos).clamp(max=50.0).detach()
        bce = F.binary_cross_entropy_with_logits(out, y, pos_weight=pw)
        p = torch.sigmoid(out)
        dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)  # whole batch: empty targets are fine
        return bce + dice, {"bce": float(bce.detach()), "dice": float(dice.detach()), "pos": float(pos)}

    @staticmethod
    def aux_target(inputs, tau=0.03):
        """Encoder inputs of the written frames [n, 1, H, W] -> [1, H, W] accumulated darkening beyond ``tau``."""
        return (-inputs.float() - float(tau)).clamp(min=0.0).amax(dim=0)

    @staticmethod
    def hint(m, size):
        """Memory field -> hint channels at the model resolution."""
        return F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)


class LearnedMemoryTrack:
    """``m = max(m, gate * E(frame))`` over a walk; ``value`` is the hint at the model resolution."""

    def __init__(self, encoder, size):
        self.encoder, self.size = encoder, int(size)
        self.field = None
        self.ref = None  # walk-start frame (delta encoder)

    @property
    def value(self):
        return None if self.field is None else MemoryEncoder.hint(self.field, self.size)

    @torch.no_grad()
    def update(self, frame, gate=True):
        if self.ref is None:
            self.ref = frame.detach().float()
        e = self.encoder(memory_input(frame, self.ref, getattr(self.encoder, "input_mode", "raw")))
        if self.field is None:
            self.field = e if gate else torch.zeros_like(e)
        elif gate:
            self.field = torch.maximum(self.field, e)
        return self.value


def _walks(dataset):
    """(rows of one (exp, direction) walk in timestep order) per walk, rows = positions in ``dataset.label``."""
    for _, rows in dataset.label.groupby(["exp", "direction"], sort=True):
        yield rows.sort_values("timestep", kind="stable")


@torch.no_grad()
def _real_frame(dataset, r):
    fr = dataset._load_frame(
        r["filename"], r["exp"], r["direction"], r["timestep"], r["DI"],
        force_flip=False, apply_random_erasing=False,
    )
    return fr["image"][None]


@torch.no_grad()
def build_memory_cache(dataset, transform, sigma=2.0, tau=0.15, window=7, gate_mode="none", ttl=0, novelty=0.0):
    """{filename: (1, H, W) float16} rule memory before that frame (zeros for the first frame of a walk),
    real frames through ``transform`` (no augmentation), unflipped."""
    saved = dataset.transform
    dataset.transform = transform
    out = {}
    try:
        for rows in _walks(dataset):
            track = MemoryTrack(sigma, tau, window, ttl, novelty if gate_mode == "novelty" else 0.0)
            gates = write_gates(rows["filename"].tolist(), gate_mode)
            for (_, r), gate in zip(rows.iterrows(), gates):
                image = _real_frame(dataset, r)
                prev = torch.zeros_like(image[:, :1]) if track.value is None else track.value
                out[str(r["filename"])] = prev[0].to(torch.float16)
                track.update(image, gate)
    finally:
        dataset.transform = saved
    print(f"[memory_cache] {dataset.mode}: {len(out)} frames (sigma {sigma}, tau {tau}, window {window}, "
          f"gate {gate_mode}, ttl {ttl}, novelty {novelty if gate_mode == 'novelty' else 0.0})", flush=True)
    return out


def build_memory_rows(dataset, offset, gate_mode="none"):
    """({filename: int64 array of the global rows of the earlier frames of its walk that write the memory},
    {filename: its own global row}); global row = ``offset`` + position in ``dataset.label``."""
    rows_of, row_of = {}, {}
    for rows in _walks(dataset):
        gates = write_gates(rows["filename"].tolist(), gate_mode)
        written = []
        for (pos, r), gate in zip(rows.iterrows(), gates):
            f = str(r["filename"])
            rows_of[f] = np.asarray(written, dtype=np.int64)
            row_of[f] = int(offset) + int(pos)
            if gate:
                written.append(int(offset) + int(pos))
    return rows_of, row_of


@torch.no_grad()
def build_memory_frames(dataset, transform, offset, gate_mode="none", input_mode="raw"):
    """Learned-encoder memory inputs. Returns (frames [N, C, H, W] float16 in ``dataset.label`` order, real,
    unaugmented, unflipped; {filename: int64 array of the global rows (``offset`` + position) of the earlier
    frames of its walk that write the memory}). Each frame is stored as ``memory_input(frame, first frame of its
    walk, input_mode)`` [1, H, W] (``raw``: the frame itself)."""
    rows_of, _ = build_memory_rows(dataset, offset, gate_mode)
    saved = dataset.transform
    dataset.transform = transform
    frames = [None] * len(dataset.label)
    try:
        for rows in _walks(dataset):
            ref = None
            for pos, r in rows.iterrows():
                image = _real_frame(dataset, r)
                ref = image if ref is None else ref
                if input_mode != "raw":
                    image = memory_input(image, ref, input_mode)
                frames[pos] = image[0].to(torch.float16)
    finally:
        dataset.transform = saved
    n_max = max(len(v) for v in rows_of.values())
    print(f"[memory_frames] {dataset.mode}: {len(frames)} frames, rows offset {offset}, up to {n_max} earlier "
          f"writes (gate {gate_mode}, input {input_mode})", flush=True)
    return torch.stack(frames), rows_of
