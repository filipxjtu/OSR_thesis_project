from pathlib import Path

from python.src.analysis import generate_dataset_figures


def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent

    raise RuntimeError("Project root not found")


def main():

    project_root = find_project_root()

    spec_version = "v2"
    seeds = [17, 27, 37]
    n_per_class = [400]

    for seed in seeds:
        for n in n_per_class:
            generate_dataset_figures(seed, n, project_root, spec_version)


if __name__ == "__main__":
    main()