from pathlib import Path
from python.src.train import train_osr_model


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Project root not found")


def main():

    project_root = find_project_root()

    # user parameters
    seeds = [32, 54, 76]
    n_per_class_list = [1000]
    model_name = "lightweight"
    spec_version = "v2"
    epochs = 50


    for seed in seeds:
        for n in n_per_class_list:

            print(f"\n=== OSR | model={model_name} seed={seed} n={n} ===")

            train_osr_model(
                model_name=model_name,
                seed=seed,
                n_per_class=n,
                spec_version=spec_version,
                project_root=project_root,
                epochs=epochs,
            )


if __name__ == "__main__":
    main()