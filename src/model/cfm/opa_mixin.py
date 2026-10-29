"""OPA-Flow extra losses (docs/redesign/phase1_opaflow.md). Everything here is off unless
``opa_lambda_ord`` / ``opa_lambda_mag`` > 0.

A frozen pixel-CNN critic R(a, b) -> (logit "b is later than a", progress gap m̂), trained in
scripts/eval/phase0_critic.py (no DINO), scores one-step predictions of the generator:

  B  image pairs. x̂₁ = z_t + (1 - t) v_θ(z_t, t),  t ~ U[t_min, 1],  z_t built from the real target
     frame. Paired with a REAL earlier frame of the same column and face (>= 1 drift level earlier):
       L_ord = softplus(-logit(a, x̂₁)),  L_mag = Huber(m̂(a, x̂₁) - (P_target - P_a)).
  C  image-free pairs (``opa_free``). Two states s < s' of a column that has no images (condition cache
     from scripts/export/export_opa_free_cache.py), the SAME initial noise, a short stop-gradient Euler
     rollout to t_s, then one differentiable step to x̂₁ for each state:
       L_ord = softplus(-logit(x̂₁(s), x̂₁(s'))),  L_mag = Huber(m̂ - (P_s' - P_s)).

P = (-strength_ratio + cum_energy) / 2 of the damage-state encoder, the same target the critic's m̂
head was fitted to. The gradient never goes through the whole ODE. The critic nets are held in a plain
object, not in an nn.Module, so checkpoints and the optimizer are unchanged.
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_critic_module():
    path = os.path.join(_ROOT, "scripts", "eval", "phase0_critic.py")
    spec = importlib.util.spec_from_file_location("opa_phase0_critic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _OPAState:
    """Frozen critics + real-frame bank + (optional) image-free condition cache."""

    def __init__(self, owner, device):
        self.mod = _load_critic_module()
        mod = self.mod
        self.device = device
        data_path = os.path.join(_ROOT, "Data", "DamagePattern", owner.opa_data_path)
        lab, imgs = mod.load_frames(
            data_path,
            os.path.join(_ROOT, "Data", "DamagePattern", "label.csv"),
            os.path.join(_ROOT, "Data", "DamagePattern", "images"),
            owner.opa_encoder_run,
        )
        self.lab = lab
        self.bank = torch.as_tensor(imgs, device=device)  # uint8 (N, 128, 128), generator preprocessing
        self.P = torch.as_tensor(lab["P"].to_numpy(), dtype=torch.float32, device=device)
        self.row_of = {str(f): i for i, f in enumerate(lab["filename"])}
        level = lab["level"].to_numpy()
        self.earlier = {}
        for _, g in lab.groupby("seq"):
            idx = g.index.to_numpy()
            for i in idx:
                cand = idx[level[idx] <= level[i] - 1]
                if len(cand):
                    self.earlier[int(i)] = cand
        self.nets = []
        for arch in owner.opa_critic_archs:
            for seed in owner.opa_critic_seeds:
                net = mod.Critic(arch)
                net.load_state_dict(torch.load(os.path.join(owner.opa_critic_dir, f"{arch}_all4_s{seed}.pt"), map_location="cpu"))
                net.eval().to(device)
                for p in net.parameters():
                    p.requires_grad_(False)
                self.nets.append(net)
        self.gen = torch.Generator(device=device).manual_seed(1234)
        self.rng = np.random.default_rng(1234)
        self.free = None
        if owner.opa_free:
            z = np.load(os.path.join(_ROOT, owner.opa_free_cache))
            # NpzFile decompresses on every access; read each array once.
            arr = {k: np.asarray(z[k]) for k in z.files}
            exp_of_row = arr["row_exp"]
            bounds = {}
            for e in np.unique(exp_of_row):
                rows = np.nonzero(exp_of_row == e)[0]
                bounds[int(e)] = (int(rows[0]), int(rows[-1]) + 1)
            self.free = dict(
                info=torch.as_tensor(arr["info"], dtype=torch.float32, device=device),
                g=torch.as_tensor(arr["g"], device=device),  # float16, cast per use
                md=arr["row_maxdrift"], P=torch.as_tensor(arr["row_P"], dtype=torch.float32, device=device),
                exp=exp_of_row, bounds=bounds,
            )
        print(
            f"[opa] {len(self.nets)} critics, bank {tuple(self.bank.shape)}, "
            f"{len(self.earlier)} frames with an earlier frame"
            + (f", free cache {len(self.free['md'])} rows / {len(self.free['bounds'])} columns" if self.free else ""),
            flush=True,
        )

    def critic(self, xa01, xb01, aug):
        """xa01, xb01: (B,1,H,W) in [0,1]. Ensemble-mean (logit that b is later, m̂)."""
        if aug:
            xa01 = self.mod.geometric(self.mod.photometric(xa01, self.gen), self.gen)
            xb01 = self.mod.geometric(self.mod.photometric(xb01, self.gen), self.gen)
        a, b = xa01 * 2 - 1, xb01 * 2 - 1
        logit = m = 0.0
        for net in self.nets:
            lo, mm = net(a, b)
            logit = logit + lo
            m = m + mm
        return logit / len(self.nets), m / len(self.nets)


class CFMOPAMixin:
    def _opa_state(self):
        st = getattr(self, "_opa_state_obj", None)
        if st is None:
            st = _OPAState(self, self.device)
            self._opa_state_obj = st
        return st

    @staticmethod
    def _opa_gray01(x):
        return ((x.mean(dim=1, keepdim=True) + 1.0) * 0.5).clamp(0.0, 1.0)

    def _opa_full_coords(self, n, like):
        gx, gy = self._build_full_coordinate_maps(self.model_spatial_size, self.model_spatial_size, device=like.device, dtype=like.dtype)
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(n, -1, -1, -1).contiguous()

    def _opa_velocity(self, z, t, c, g, cmask, coords):
        return self._predict_velocity(
            z, t, c, coords=coords, damage_mask=cmask, gru_feat=g,
            self_cond=torch.zeros_like(z) if self.use_self_conditioning else None,
        )

    def _opa_losses(self, logit, m, dP):
        l_ord = F.softplus(-logit).mean()
        l_mag = F.smooth_l1_loss(m, dP, beta=0.5)
        return l_ord, l_mag

    # ------------------------------------------------------------------------------------ group B
    def _opa_image_pair_losses(self, z1, c, gru_feat, condition_mask):
        meta = getattr(self, "_opa_meta", None)
        if meta is None:
            raise RuntimeError("opa: batch has no filename (dataset needs opa_lambda_* > 0 in train mode)")
        st = self._opa_state()
        names, flipped = meta
        rows = [st.row_of.get(str(n), -1) for n in names]
        valid = [i for i, r in enumerate(rows) if r in st.earlier]
        if not valid:
            return None
        pick = st.rng.permutation(valid)[: self.opa_n]
        sel = torch.as_tensor(pick, device=z1.device)
        tgt_rows = [rows[i] for i in pick]
        a_rows = [int(st.rng.choice(st.earlier[r])) for r in tgt_rows]
        flip = torch.as_tensor(flipped, device=z1.device)[sel].bool()

        n = len(pick)
        zt1 = z1[sel]
        z0 = torch.randn_like(zt1)
        t = self.opa_t_min + (1.0 - 1e-3 - self.opa_t_min) * torch.rand(n, device=z1.device)
        tb = t.view(-1, 1, 1, 1)
        zt = (1.0 - tb) * z0 + tb * zt1
        cm = None if condition_mask is None else self._align_spatial_to_target(condition_mask)[sel]
        coords = self._opa_full_coords(n, zt)
        v = self._opa_velocity(zt, t, c[sel], None if gru_feat is None else gru_feat[sel], cm, coords)
        xhat = zt + (1.0 - tb) * v  # grad flows through v only

        xa = st.bank[torch.as_tensor(a_rows, device=z1.device)].float().unsqueeze(1) / 255.0
        xa = torch.where(flip.view(-1, 1, 1, 1), xa.flip(-1), xa)
        logit, m = st.critic(xa, self._opa_gray01(xhat), self.opa_aug)
        dP = st.P[torch.as_tensor(tgt_rows, device=z1.device)] - st.P[torch.as_tensor(a_rows, device=z1.device)]
        l_ord, l_mag = self._opa_losses(logit, m, dP)
        with torch.no_grad():
            self._opa_stats = {
                "opa/img_acc": float((logit > 0).float().mean()), "opa/img_m": float(m.mean()),
                "opa/img_dP": float(dP.mean()), "opa/img_n": float(n),
            }
        return l_ord, l_mag

    # ------------------------------------------------------------------------------------ group C
    def _opa_free_pair_losses(self, ref):
        st = self._opa_state()
        fr = st.free
        dev = ref.device
        n = self.opa_free_n
        exps = list(fr["bounds"].keys())
        ia, ib = [], []
        guard = 0
        while len(ia) < n and guard < 200 * n:
            guard += 1
            e = exps[int(st.rng.integers(len(exps)))]
            lo, hi = fr["bounds"][e]
            md = fr["md"][lo:hi]
            s = int(st.rng.integers(len(md)))
            j0 = int(np.searchsorted(md, md[s] * self.opa_free_ratio, side="left"))
            if j0 >= len(md):
                continue
            ia.append(lo + s)
            ib.append(lo + int(st.rng.integers(j0, len(md))))
        if not ia:
            return None
        ia, ib = np.asarray(ia), np.asarray(ib)
        n = len(ia)
        rows = np.concatenate([ia, ib])
        info = fr["info"][torch.as_tensor(fr["exp"][rows], device=dev)]
        g = fr["g"][torch.as_tensor(rows, device=dev)].float()
        face = torch.as_tensor(st.rng.integers(0, 2, size=n), device=dev)
        d1 = F.one_hot(face, 2).float()
        direction = torch.cat([d1, d1], 0)  # same face for both states of a pair
        DI = torch.zeros(2 * n, 1, device=dev)
        c, gru_feat = self._unpack_conditions(info, g, direction, DI)
        z0 = torch.randn(n, 3, self.model_spatial_size, self.model_spatial_size, device=dev)
        z = torch.cat([z0, z0], 0)  # shared start noise
        ts = (0.5 + 0.25 * torch.rand(n, device=dev)).repeat(2)
        cm = torch.ones(2 * n, 1, z.shape[-2], z.shape[-1], device=dev) if self.use_damage_mask_condition else None
        coords = self._opa_full_coords(2 * n, z)
        k = self.opa_free_roll_steps
        with torch.no_grad():
            for i in range(k):
                tau = ts * (i / k)
                z = z + (ts / k).view(-1, 1, 1, 1) * self._opa_velocity(z, tau, c, gru_feat, cm, coords)
        v = self._opa_velocity(z, ts, c, gru_feat, cm, coords)
        xhat = z + (1.0 - ts).view(-1, 1, 1, 1) * v
        x = self._opa_gray01(xhat)
        logit, m = st.critic(x[:n], x[n:], self.opa_aug)
        dP = fr["P"][torch.as_tensor(ib, device=dev)] - fr["P"][torch.as_tensor(ia, device=dev)]
        l_ord, l_mag = self._opa_losses(logit, m, dP)
        with torch.no_grad():
            self._opa_stats.update({
                "opa/free_acc": float((logit > 0).float().mean()), "opa/free_m": float(m.mean()),
                "opa/free_dP": float(dP.mean()), "opa/free_n": float(n),
            })
        return l_ord, l_mag

    # --------------------------------------------------------------------------------------- total
    def _opa_extra_loss(self, z1, c, gru_feat, condition_mask, loss_fm=None):
        self._opa_stats = {}
        terms = []  # (name, weight, tensor)
        img = self._opa_image_pair_losses(z1, c, gru_feat, condition_mask)
        if img is not None:
            terms += [("img_ord", self.opa_lambda_ord, img[0]), ("img_mag", self.opa_lambda_mag, img[1])]
        if self.opa_free:
            fre = self._opa_free_pair_losses(z1)
            if fre is not None:
                w = self.opa_free_weight
                terms += [("free_ord", w * self.opa_lambda_ord, fre[0]), ("free_mag", w * self.opa_lambda_mag, fre[1])]
        if not terms:
            return None
        total = sum(w * v for _, w, v in terms)
        for name, _, v in terms:
            self.log(f"opa/{name}", v.detach(), on_step=True, on_epoch=False)
        for k, v in self._opa_stats.items():
            self.log(k, v, on_step=True, on_epoch=False)
        self.log("opa/total", total.detach(), on_step=True, on_epoch=False)
        every = int(getattr(self, "opa_diag_every", 0))
        if every > 0 and loss_fm is not None and self.global_step % every == 0:
            self._opa_gradnorm_diag(loss_fm, terms)
        return total

    def _opa_gradnorm_diag(self, loss_fm, terms):
        """Gradient norm of each term w.r.t. the velocity net (diagnostic for choosing λ)."""
        params = [p for p in self.velocity_net.parameters() if p.requires_grad]
        out = {}
        for name, v in [("fm", loss_fm)] + [(n, t) for n, _, t in terms]:
            if not v.requires_grad:
                continue
            gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            out[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        print(f"[opa-diag] step {self.global_step} grad-norm per unit weight: "
              + " ".join(f"{k}={v:.3g}" for k, v in out.items()), flush=True)
