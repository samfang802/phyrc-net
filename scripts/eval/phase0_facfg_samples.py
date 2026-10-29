#!/usr/bin/env python3
"""generate_eval_samples.py with FA-CFG's "undamaged" g0 set per sample (phase-0 guidance check).

FA-CFG (docs/guidance.md §3) needs `cfm._facfg_gru_unc`: the GRU condition of the lowest-DI frame of
the same column and direction (as export_di_tertile_overview_grids.py `--facfg_g0 undamaged`). The
eval sampler has no hook for a per-sample tensor, so this wrapper runs generate_eval_samples.main()
unchanged and only wraps its batch sampler to set that tensor for the rows of each batch
(shuffle=False, so rows are in label order). Same CLI as generate_eval_samples.py; FA-CFG weights go
through --set, e.g.

  python scripts/eval/phase0_facfg_samples.py --cfm_result_dir ... --cfm_checkpoint ... --split test \
      --k 8 --out gn_facfg.npz --set facfg_wl=1.0 facfg_wh=2.0

Only the single-frame, non-cascade path is supported.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_spec = importlib.util.spec_from_file_location("_gen_eval", os.path.join(_ROOT, "scripts", "eval", "generate_eval_samples.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

STATE = {"row": 0}
_orig_ds = gen.PatternLoopDataset
_orig_batch = gen._eval_r2._generate_batch


def _dataset(*a, **kw):
    ds = _orig_ds(*a, **kw)
    lab = ds.label.reset_index(drop=True)
    H = torch.stack([torch.as_tensor(ds.hidden_state[e][int(t), :], dtype=torch.float32)
                     for e, t in zip(lab["exp"], lab["timestep"])])
    und = np.empty(len(lab), dtype=np.int64)
    for _, grp in lab.groupby(["exp", "direction"], sort=False):
        anchor = grp.sort_values(["DI", "timestep"]).index[0]
        und[grp.index.to_numpy()] = anchor
        print(f"[facfg-g0] {grp['exp'].iloc[0]}/{grp['direction'].iloc[0]}: g0 = {lab.loc[anchor, 'filename']} "
              f"(DI {lab.loc[anchor, 'DI']:.4f})", flush=True)
    STATE.update(ds=ds, H=H, und=und, n=len(lab))
    return ds


def _batch(cfm, info, hidden_state, direction, DI, damage_mask=None):
    if bool(getattr(cfm, "use_cascade_sr", False)):
        raise ValueError("phase0_facfg_samples.py: cascade sampler not supported")
    if getattr(cfm, "facfg_wl", None) is None or getattr(cfm, "facfg_wh", None) is None:
        raise ValueError("set facfg_wl and facfg_wh with --set")
    r, b = STATE["row"], hidden_state.shape[0]
    rows = np.arange(r, r + b)
    # Row tracking check: the batch condition must be the one of these label rows.
    if not torch.allclose(hidden_state.detach().float().cpu().reshape(b, -1), STATE["H"][rows].reshape(b, -1), atol=1e-5):
        raise RuntimeError(f"row tracking lost at row {r}")
    g0 = STATE["H"][STATE["und"][rows]].to(hidden_state.device, hidden_state.dtype)
    cfm._facfg_gru_unc = cfm.process_physics_condition(g0)
    STATE["row"] = (r + b) % STATE["n"]
    return _orig_batch(cfm, info, hidden_state, direction, DI, damage_mask=damage_mask)


gen.PatternLoopDataset = _dataset
gen._eval_r2._generate_batch = _batch

if __name__ == "__main__":
    if "--shared_noise" in sys.argv:
        raise SystemExit("--shared_noise path does not go through the wrapped sampler")
    gen.main()
