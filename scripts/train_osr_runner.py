from pathlib import Path

import torch

from python.src.train import train_osr_model
from python.src.train.osr_hparams import OSRHParams


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Project root not found")


def main():
    project_root = find_project_root()

    seeds = [216]
    n_per_class_list = [2500]
    spec_version = "v2"
    epochs = 50

    hparams = OSRHParams()

    for seed in seeds:
        for n in n_per_class_list:
            print(f"\n=== OSR Training | seed={seed} | n={n} ===")

            trained_model = train_osr_model(
                seed=seed,
                n_per_class=n,
                spec_version=spec_version,
                project_root=project_root,
                epochs=epochs,
                hparams=hparams,
            )

            del trained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()