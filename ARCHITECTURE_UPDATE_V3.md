# ARCHITECTURE_UPDATE_V3

## 1. Executive Summary

This V3 update consolidates the RC bridge column damage-generation pipeline into an execution-ready baseline centered on **CFM Stage 3 full-dataset training** with stable condition wiring and reproducible Colab workflow packaging.

Compared with V2 (architecture transition plan), V3 focuses on implementation hardening:

- **Extractor onboarding is now operationalized**: `best.ckpt` + `args.json` are prepared automatically for Stage 3.
- **Notebook flow is re-packaged** into ordered cells with a direct training path from environment setup to visualization.
- **Condition pathway remains fixed and explicit** (`c = [hidden_state, direction, DI]`) for repeatable experiments and ablations.

From a research perspective, V3 shifts from conceptual migration to reproducible experimentation. From an engineering perspective, it reduces setup fragility, removes manual mismatch points, and shortens handoff time.

---

## 2. Core Architecture Evolution

### Sequence Conditioning: Phase-1 GRU Latent as Stable Interface

The current production path keeps the Stage-1 extractor interface stable through `HystereticGRU` outputs.

- **Condition source**: latent hidden state extracted from the trained Phase-1 checkpoint.
- **Why keep GRU interface in V3**: compatibility with existing datasets/checkpoints and immediate reproducibility.
- **Engineering effect**: Stage 3 can run independently as long as extractor bundle is valid.

> V3 explicitly prioritizes reliability and continuity of experiments over introducing additional sequence backbone changes in the same milestone.

### Generative Backend: CFM + ADM U-Net (Operational Baseline)

The Stage-3 generator remains CFM-based and learns a conditional velocity field:

- $Z_0 \sim \mathcal{N}(0, I)$
- $Z_1$: target damage image
- $Z_t = (1-t)Z_0 + tZ_1$
- $u_t = Z_1 - Z_0$
- Minimize flow-matching objective between $v_\theta(Z_t, t, c)$ and $u_t$

Condition injection follows the same CFMADM path and preserves conditioning controllability while avoiding discriminator instability from GAN-centric training.

### Inference: Heun ODE Sampling with Periodic Monitoring

Sampling is performed via ODE integration using Heun solver.

- **Default solver**: `heun`
- **Monitoring enhancement**: periodic inference image saving/logging every configurable N epochs
- **Practical effect**: better visibility of training dynamics and earlier detection of condition drift/failure modes

---

## 3. Tensor Dimension Transformations (CRITICAL)

Canonical V3 shape flow:

```text
Phase-1 sequence branch:
Input loop sequence:                 [B, L, C_seq]
-> GRU extractor output:             [B, L, D_latent]
-> timestep selection / indexing:    [B, D_latent]

Condition branch:
Direction one-hot:                   [B, 2]
Damage index (DI):                   [B, 1]
Condition concat c:                  [B, D_cond]

CFM branch:
Target image Z1:                     [B, C_img, H, W]
Noise Z0:                            [B, C_img, H, W]
Interpolated Zt:                     [B, C_img, H, W]
Time t:                              [B]

Velocity network:
(Zt, t, c) -> v_theta:               [B, C_img, H, W]
Target velocity u_t = Z1 - Z0:       [B, C_img, H, W]
```

Condition composition variants (same API):

- **Latent-only**: $c = [h]$
- **Latent + direction**: $c = [h, d]$
- **Latent + direction + DI (V3 default)**: $c = [h, d, \mathrm{DI}]$

---

## 4. Loss Functions & Optimization Strategy

### Primary Objective: Flow Matching

V3 baseline optimizes flow-matching as the primary objective:

- $\mathcal{L}_{FM} = \|v_\theta(Z_t, t, c) - (Z_1 - Z_0)\|_2^2$

Optional DI-related auxiliary branches can be enabled, but default full-dataset CFM runs keep configuration simple and stable.

### Optimizer and Scheduler

- **Optimizer**: AdamW (default in current training stack)
- **Scheduler**: warmup + cosine decay (`warmup_cosine`)
- **V3 full-run defaults**:
  - `epoch=1000`
  - `warmup_epochs=100`
  - `min_lr_ratio=0.1`
  - `batch_size=4` (Colab-safe baseline)
  - `num_workers=0` (to minimize environment-dependent dataloader issues)

### Validation and Inference Cadence

V3 introduces explicit training cadence controls in Stage 3:

- `val_every_n_epoch=10`
- `inference_every_n_epochs=10`
- `inference_num_images=6`

This improves observability and comparability across long runs.

---

## 5. Paradigm Shift in Evaluation Metrics

V3 keeps practical image metrics for day-to-day training diagnostics while preserving room for representation-aware evaluation:

- **Operational metrics**: PSNR, SSIM, FID (in notebook validation workflow)
- **Known interpretation rule**: low flow loss does not always guarantee low FID
- **Research direction retained**: semantic/topological metrics (e.g., DINO-style checks) remain recommended for future benchmark-quality claims

V3 emphasis is reproducible monitoring and consistent metric logging under a fixed condition path.

---

## 6. Pseudocode (Training and Inference)

### Extractor Bundle Preparation (V3 Packaging Step)

```text
Given project root:
  If phase1_extractor/best.ckpt missing and root best.ckpt exists:
    copy root best.ckpt -> phase1_extractor/best.ckpt

  If phase1_extractor/args.json missing:
    load ckpt = torch.load(best.ckpt)
    hp = ckpt["hyper_parameters"]
    args_payload = defaults + hp
    if signal_feat absent:
      infer from state_dict["L.gru.weight_ih_l0"].shape[1]
    save args_payload -> phase1_extractor/args.json
```

### Stage-3 CFM Full Training (Packaged Command Build)

```text
TRAIN_CFG = {
  architecture="CFMADM",
  cfm_solver="heun",
  cfm_ode_steps=50,
  extractor_path=resolved_path,
  latent_conditioning=True,
  direction_conditioning=True,
  include_di_conditioning=True,
  ...
}

cmd = ["python", "train_mixed.py"] + flatten(TRAIN_CFG)
subprocess.run(cmd, check=True)
```

### Inference with Heun ODE

```text
Given condition c [B, D_cond], steps N:
  z <- Gaussian noise
  for i in 0..N-1:
    t_i <- i / N
    v_i <- v_theta(z, t_i, c)
    z_pred <- z + dt * v_i
    v_next <- v_theta(z_pred, t_{i+1}, c)
    z <- z + 0.5 * dt * (v_i + v_next)
  return z
```

---

## 7. Deprecations

The following remain deprecated for new baseline claims:

- Discriminator-centric SGAN/RSGAN/RaSGAN as primary generation path
- Legacy disconnected trial scripts not aligned with current CFM training/inference flow
- Manual extractor setup steps that rely on undocumented checkpoint assumptions

Policy:

- Keep archival compatibility when needed.
- Use V3 packaged CFM workflow for all new reproducible runs.

---

## 8. Known Issues & Future Work

### Known Issues

- **Checkpoint metadata dependency**:
  - automatic `args.json` generation requires valid `hyper_parameters` in `best.ckpt`
- **Metric dependency**:
  - FID requires `torch-fidelity`
- **Runtime trade-off**:
  - higher ODE steps improve quality but increase inference/training validation cost

### Future Work

- Integrate adaptive ODE step scheduling (quality/speed balance)
- Add robust fallback when `hyper_parameters` is missing (template + strict validator)
- Expand validation to multi-sample inference panels and aggregated reports
- Re-open sequence backbone upgrades (e.g., Mamba) after V3 baseline stabilization

---

## Closing Note

V3 marks the transition from architecture intent to a reproducible training package: extractor preparation is automated, Stage-3 CFM training is parameterized and ordered, and validation is integrated into a consistent Colab-first execution flow. This provides a stable baseline for both immediate experiments and future model upgrades.
