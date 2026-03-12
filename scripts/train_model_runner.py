from pathlib import Path

from python.src.train import train_model

def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")


def main():

    project_root = find_project_root()

    models = ["first_residual_cnn"]
    spec_version = "v2"
    seeds = [22, 32, 40]
    n_per_class = [400]

    for m in models:
        for s in seeds:
            for n in n_per_class:

                print(f"\n\nRunning experiment model = {m}, seed={s}, n per class = {n}")
                print("==========================================================================\n")

                train_model(seed=s, project_root=project_root, model_name=m, n_per_class=n, spec_version=spec_version)


if __name__ == "__main__":
    main()