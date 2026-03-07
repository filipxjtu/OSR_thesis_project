from pathlib import Path

from python.src.train.model_trainer import train_model

def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")


def main():

    project_root = find_project_root()

    model = "baseline_cnn"

    seeds = [70, 123]

    for seed in seeds:

        print(f"\n\nRunning experiment seed={seed}")
        print("===================================\n")

        train_model(seed=seed, project_root=project_root, model_name=model)


if __name__ == "__main__":
    main()