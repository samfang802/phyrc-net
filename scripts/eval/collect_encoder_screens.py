#!/usr/bin/env python3
"""Mean ± std over seeds of the encoder screens (scripts/eval/screen_condition_encoder.py) per sweep tag.

Selection columns use only the four train specimens (leave-one-specimen-out) and R307 (val); CTR1 (test)
is reported but must not be used to choose a configuration.

  python scripts/eval/collect_encoder_screens.py results/DamageStateEncoder/<tag> [<tag> ...]
"""

import glob, json, os, sys

import numpy as np
import pandas as pd

SELECT = ["R_spall LOSO RMSE (train 4)", "DI LOSO RMSE (train 4)", "R_spall R307 6%", "R_spall R307 8%", "DI R307 6%", "DI R307 8%"]
TEST = ["R_spall CTR1 6%", "R_spall CTR1 8%", "DI CTR1 6%", "DI CTR1 8%"]


def main():
    rows, gru = [], None
    for tag_dir in sys.argv[1:]:
        tag = os.path.basename(tag_dir.rstrip("/"))
        vals = []
        for mfile in sorted(glob.glob(os.path.join(tag_dir, "2026-*", "screen", "metrics.json"))):
            m = json.load(open(mfile))
            args = json.load(open(os.path.join(os.path.dirname(os.path.dirname(mfile)), "args.json")))
            rep = "encoder [z_inv, s]" if args.get("input_mode", "tokens") == "points" else "encoder [z, s]"
            vals.append({**m[rep], "force_rmse": m["_force_rmse_median"]["encoder"]})
            gru = gru or {**m["GRU g"], "force_rmse": m["_force_rmse_median"]["GRU"]}
        if not vals:
            continue
        d = pd.DataFrame(vals)
        r = {"config": tag, "seeds": len(d)}
        for c in ["sign_var", *SELECT, *TEST, "force_rmse"]:
            r[c] = f"{d[c].mean():+.3f} ± {d[c].std(ddof=0):.3f}"
        # selection score: mean |error| over the selection columns (lower is better)
        r["select_score"] = f"{d[SELECT].abs().mean(axis=1).mean():.3f} ± {d[SELECT].abs().mean(axis=1).std(ddof=0):.3f}"
        rows.append(r)
    if gru:
        g = {"config": "GRU g", "seeds": 1, **{c: f"{gru[c]:+.3f}" for c in ["sign_var", *SELECT, *TEST, "force_rmse"]}}
        g["select_score"] = f"{np.mean([abs(gru[c]) for c in SELECT]):.3f}"
        rows.insert(0, g)
    print(pd.DataFrame(rows).set_index("config").T.to_string())


if __name__ == "__main__":
    main()
