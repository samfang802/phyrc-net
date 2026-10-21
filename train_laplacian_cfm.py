"""Train one Laplacian-band CFM (D1 / D2 / D3). Independent of the other two.

Usage::

    python train_laplacian_cfm.py --laplacian_band d1 ...   # same flags as train_mixed.py
    python train_laplacian_cfm.py --laplacian_band d2 ...
    python train_laplacian_cfm.py --laplacian_band d3 ...

The three-job PBS wrapper (job/DamagePatternGeneration/train_laplacian_cfm.sh) was removed in the job cleanup;
recover it with ``git show d635bf6:job/DamagePatternGeneration/train_laplacian_cfm.sh``.
"""

import sys

# Force architecture before train_mixed parses argv.
if "--architecture" not in sys.argv:
    sys.argv.extend(["--architecture", "LaplacianBandCFM"])
if "--loss_type" not in sys.argv:
    sys.argv.extend(["--loss_type", "LaplacianCFM"])

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "train_mixed.py"), run_name="__main__")
