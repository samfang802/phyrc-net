"""Zero-training texture refinement of existing samples with a pretrained ImageNet ADM prior.

Inverse-problem view (DDS-style, as in the nycu-project CFM prior): the observation is the coarse
layout of our own sample, y = A x_ours with A = block average by ``--factor`` (on the 128 grid), so the
spall layout and amount stay those of the conditional model. The pretrained OpenAI guided-diffusion ADM
(ImageNet, eps-prediction, VP schedule, no fine-tuning) supplies the detail inside each block.

The sampler runs in the project's CFM convention (OT path, t = 0 noise, t = 1 data):
  z = x_VP / (sqrt(abar) + sqrt(1 - abar)),  t = sqrt(abar) / (sqrt(abar) + sqrt(1 - abar)),
  so z = (1 - t) eps + t x1. Each step:

  x1_hat = (x_VP - sqrt(1 - abar) eps_hat) / sqrt(abar)    (ADM queried at the continuous VP index of t)
  x1_hat <- gray(x1_hat)                                   (images are grayscale)
  x1_hat <- x1_hat + lam * U(y - A x1_hat)                 (closed-form data consistency; U = nearest
                                                            upsample = pseudo-inverse of block average)
  v = (x1_hat - z) / (1 - t);  z <- z + dt v               (Euler; --heun for Heun)

With Euler this equals DDIM (eta = 0) with the implied noise re-derived after the data-consistency
step, i.e. DDS. Start: z = (1 - s) x_ours + s eps with s = ``--strength`` (the OT noise fraction, as
``noise_frac`` in ode_mixin._bridge_correct_frame). The start noise is shared by all rows with the
same k (seed + k), so a trajectory's frames get the same noise, as with ``--shared_noise`` in
generate_eval_samples.py.

Reads and writes the src/metrics/samples_io.py npz layout; only ``fake`` and ``meta`` change.

Example:
  python3 scripts/eval/imagenet_prior_refine.py --in_npz results/eval_suite/v11app_refiner/test/samples/app_r_6k.npz \
      --out_npz results/eval_suite/imagenet_prior/test/samples/inp_u256_f4_s05.npz --model u256 --factor 4 --strength 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GD_ROOT = os.path.join(ROOT, "third_party", "guided-diffusion")
sys.path.insert(0, GD_ROOT)
sys.path.insert(0, ROOT)

from src.metrics.samples_io import save_samples_npz  # noqa: E402

# Flags from the guided-diffusion README.
MODELS = {
    "u256": dict(
        ckpt="256x256_diffusion_uncond.pt", image_size=256, class_cond=False,
        num_channels=256, num_head_channels=64, num_heads=4, num_res_blocks=2,
        attention_resolutions="32,16,8", resblock_updown=True, use_scale_shift_norm=True,
        learn_sigma=True, noise_schedule="linear", diffusion_steps=1000,
    ),
    "c128": dict(
        ckpt="128x128_diffusion.pt", image_size=128, class_cond=True,
        num_channels=256, num_heads=4, num_head_channels=-1, num_res_blocks=2,
        attention_resolutions="32,16,8", resblock_updown=True, use_scale_shift_norm=True,
        learn_sigma=True, noise_schedule="linear", diffusion_steps=1000,
    ),
}


def load_model(name, device, fp16):
    # Imported here so the sampler math can be tested without the third-party package.
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    spec = dict(MODELS[name])
    ckpt = spec.pop("ckpt")
    opts = model_and_diffusion_defaults()
    opts.update(spec)
    opts["use_fp16"] = bool(fp16)
    model, diffusion = create_model_and_diffusion(**opts)
    state = torch.load(os.path.join(GD_ROOT, "models", ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if fp16:
        model.convert_to_fp16()
    model.to(device).eval()
    abar = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    return model, abar, spec["image_size"]


def block_mean(x, f):
    return F.avg_pool2d(x, f) if f > 1 else x


def block_up(y, f):
    return F.interpolate(y, scale_factor=f, mode="nearest") if f > 1 else y


class OTSchedule:
    """Map an OT time t (0 noise, 1 data) to the ADM's continuous VP index and its abar."""

    def __init__(self, abar):
        a = abar.double()
        self.sa, self.sn = a.sqrt(), (1 - a).sqrt()
        self.t_of_k = (self.sa / (self.sa + self.sn)).numpy()  # decreasing in k
        self.t_min, self.t_max = float(self.t_of_k[-1]), float(self.t_of_k[0])
        self.k = np.arange(len(a), dtype=np.float64)

    def at(self, t):
        """(continuous index k, sqrt(abar), sqrt(1 - abar)) for OT time t, linear in k between grid points."""
        t = float(np.clip(t, self.t_min, self.t_max))
        k = float(np.interp(t, self.t_of_k[::-1], self.k[::-1]))
        k0 = int(np.floor(k)); k1 = min(k0 + 1, len(self.k) - 1); w = k - k0
        sa = float((1 - w) * self.sa[k0] + w * self.sa[k1])
        sn = float((1 - w) * self.sn[k0] + w * self.sn[k1])
        return k, sa, sn


def make_denoiser(model, sched, y, f, lam, class_label):
    """x1_hat(z, t) with gray projection and closed-form data consistency on the block average."""
    kw = {}
    if class_label is not None:
        kw["y"] = torch.full((y.shape[0],), int(class_label), device=y.device, dtype=torch.long)

    def x1_hat(z, t):
        k, sa, sn = sched.at(t)
        x_vp = z * (sa + sn)
        kk = torch.full((z.shape[0],), k, device=z.device, dtype=torch.float32)
        eps = model(x_vp, kk, **kw)[:, :3].float()
        x1 = ((x_vp - sn * eps) / sa).clamp(-1, 1).mean(dim=1, keepdim=True)
        if lam > 0:
            x1 = (x1 + lam * block_up(y - block_mean(x1, f), f)).clamp(-1, 1)
        return x1

    return x1_hat


@torch.no_grad()
def refine_batch(model, sched, x_ours, eps0, ts, factor, lam, class_label, img_size, heun=False):
    """x_ours: (B, 1, 128, 128) in [-1, 1]. eps0: (B, 3, S, S). ts: increasing OT times. Returns (B, 1, 128, 128)."""
    x = x_ours
    if img_size != x.shape[-1]:
        x = F.interpolate(x, size=(img_size, img_size), mode="bicubic", align_corners=False).clamp(-1, 1)
    f = factor * img_size // x_ours.shape[-1]
    den = make_denoiser(model, sched, block_mean(x, f), f, lam, class_label)
    z = ts[0] * x.expand(-1, 3, -1, -1) + (1 - ts[0]) * eps0
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        x1 = den(z, t)
        v = (x1 - z) / (1 - t)
        if heun and i + 2 < len(ts):  # last interval stays Euler: 1 - tn is ~0 there
            z_e = z + (tn - t) * v
            v2 = (den(z_e, tn) - z_e) / (1 - tn)
            z = z + (tn - t) * 0.5 * (v + v2)
        else:
            z = z + (tn - t) * v
    x1 = den(z, ts[-1])
    if img_size != x_ours.shape[-1]:
        x1 = F.interpolate(x1, size=x_ours.shape[-2:], mode="area")
    return x1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_npz", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--model", choices=sorted(MODELS), default="u256")
    p.add_argument("--factor", type=int, default=4, help="block size of the observation on the 128 grid")
    p.add_argument("--strength", type=float, default=0.5, help="OT noise fraction of the start: z = (1-s) x + s eps")
    p.add_argument("--steps", type=int, default=50, help="steps for strength 1; scaled by strength")
    p.add_argument("--heun", action="store_true", help="Heun instead of Euler (about 2x the model calls)")
    p.add_argument("--lam", type=float, default=1.0, help="data-consistency weight; 0 = plain SDEdit")
    p.add_argument("--class_label", type=int, default=825, help="ImageNet class for c128 (825 = stone wall)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    p.add_argument("--k_limit", type=int, default=0, help="first K samples per row only (smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", type=int, default=-1, help="-1 = on for cuda")
    args = p.parse_args()

    if 128 % args.factor:
        raise ValueError(f"--factor must divide 128, got {args.factor}")
    if not (0.0 < args.strength <= 1.0):
        raise ValueError(f"--strength must be in (0, 1], got {args.strength}")
    fp16 = (args.device.startswith("cuda")) if args.fp16 < 0 else bool(args.fp16)

    d = np.load(args.in_npz, allow_pickle=False)
    fake = d["fake"]
    n_rows = args.limit or fake.shape[0]
    n_k = args.k_limit or fake.shape[1]
    fake = fake[:n_rows, :n_k]
    N, K, H, W = fake.shape

    model, abar, img_size = load_model(args.model, args.device, fp16)
    sched = OTSchedule(abar)
    t_start = max(1.0 - args.strength, sched.t_min)
    n_steps = max(2, int(round(args.steps * args.strength)))
    ts = [float(t) for t in np.linspace(t_start, sched.t_max, n_steps + 1)]
    class_label = args.class_label if MODELS[args.model]["class_cond"] else None

    out = np.empty_like(fake)
    start = time.time()
    for k in range(K):
        g = torch.Generator(device="cpu").manual_seed(args.seed + k)
        eps_k = torch.randn((1, 3, img_size, img_size), generator=g)
        for r0 in range(0, N, args.batch_size):
            r1 = min(N, r0 + args.batch_size)
            xb = torch.from_numpy(fake[r0:r1, k]).float().div(127.5).sub(1.0).unsqueeze(1).to(args.device)
            eb = eps_k.expand(r1 - r0, -1, -1, -1).to(args.device)
            yb = refine_batch(model, sched, xb, eb, ts, args.factor, args.lam, class_label, img_size, args.heun)
            out[r0:r1, k] = ((yb.squeeze(1).clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
        print(f"[imagenet_prior] k {k + 1}/{K} done, {time.time() - start:.0f}s", flush=True)

    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    meta = {"source": meta, "source_npz": args.in_npz, "imagenet_prior": vars(args) | {"timesteps": ts}}
    save_samples_npz(
        args.out_npz, out, d["real"][:N], d["cond"][:N], d["DI"][:N], d["filename"][:N], meta=meta,
    )
    diff = np.abs(out.astype(np.float32) - fake.astype(np.float32))
    print(f"[imagenet_prior] wrote {args.out_npz}: {N} rows x {K}, {len(ts) - 1} steps, t {ts[0]:.3f} -> {ts[-1]:.3f}; "
          f"mean |out - in| {diff.mean():.2f} grey levels", flush=True)


if __name__ == "__main__":
    main()
