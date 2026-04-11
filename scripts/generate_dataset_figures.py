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
    seeds = [321, 456, 987]
    n_per_class_list = [1000]

    for seed in seeds:
        for n_per_class in n_per_class_list:
            generate_dataset_figures(
                seed=seed,
                n_per_class=n_per_class,
                project_root=project_root,
                spec_version=spec_version,
            )


if __name__ == "__main__":
    main()