from pathlib import Path
from python.src.analysis.dataset_figures import generate_dataset_figures


def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent

    raise RuntimeError("Project root not found")


def main():

    project_root = find_project_root()

    seeds = [22, 42, 121]

    for seed in seeds:
        generate_dataset_figures(seed, project_root)


if __name__ == "__main__":
    main()