from pathlib import Path

import torch

from python.src.train import train_model

def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")


def main():

    project_root = find_project_root()

    models = ["ts_ms_va_drsn"]
    spec_version = "v2"
    seeds = [32]
    n_per_class = [1000]
    epochs = 30

    for m in models:
        for s in seeds:
            for n in n_per_class:

                print(f"\n\nRunning experiment model = {m}, seed={s}, n per class = {n}")
                print("==========================================================================\n")

                trained_model = train_model(
                    seed=s,
                    project_root=project_root,
                    model_name=m,
                    n_per_class=n,
                    spec_version=spec_version,
                    n_epochs=epochs,
                )
                # clean up RAM
                del trained_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()



if __name__ == "__main__":
    main()