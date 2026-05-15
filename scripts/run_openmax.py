"""
scripts/run_openmax.py
=======================
Fits the OpenMax head on top of an already-trained AsymmetricTriNet checkpoint.

Pipeline (delegated to train_openmax_model):
  1. Load closed-set asymmetric_trinet checkpoint (frozen backbone).
  2. Compute per-class MAV from training AVs.
  3. Fit per-class Weibull tails.
  4. Calibrate global rejection threshold on val knowns + proxy unknowns.
  5. Save openmax_trinet_seed{seed}_n{n}.pt + JSON log.

Run from project root:
    python scripts/run_openmax.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from python.src.train import train_openmax_model, OpenMaxHParams


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 146
N_PER_CLASS  = 2500
SPEC_VERSION = "v2"

# OpenMax hyperparameters (tweak here if needed)
HPARAMS = OpenMaxHParams(
    alpha_rank=10,
    tail_size=20,
    distance="euclidean",
    only_correct=True,
    fpr_cap=0.4,            # matches OsrSAF default for fair comparison
    batch_size=32,
)


def main():
    print(f"\n\nRunning OpenMax fit | seed = {SEED}, n_per_class = {N_PER_CLASS}")
    print("=" * 78)

    ckpt_path = PROJECT_ROOT / "artifacts" / "checkpoints" / f"openmax_trinet_seed{SEED}_n{N_PER_CLASS}.pt"
    if ckpt_path.exists():
        print(f"[SKIP] OpenMax already fit: {ckpt_path.name}")
        print("       Delete the checkpoint above to refit.")
        return

    train_openmax_model(
        seed=SEED,
        n_per_class=N_PER_CLASS,
        spec_version=SPEC_VERSION,
        project_root=PROJECT_ROOT,
        hparams=HPARAMS,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("OpenMax fit complete.\n")


if __name__ == "__main__":
    main()