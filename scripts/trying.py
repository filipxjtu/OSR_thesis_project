from python.src.dataio import load_artifact
from pathlib import Path

def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "artifacts").exists():
            return parent
    raise RuntimeError("Could not locate thesis_project root.")


spec_version = "v2"
seed = 216
n_per_class = 2500

project_root = find_project_root()
dataset_dir = project_root / "artifacts" / "datasets"
train_file = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_train.mat"
eval_file  = dataset_dir / "impaired" / f"impaired_dataset_{spec_version}_seed{seed}_n{n_per_class}_eval.mat"


train_artifact = load_artifact(str(train_file), load_params=False)
#eval_artifact = load_artifact(str(eval_file), load_params=False)

print(train_artifact.X[1:5])

print(train_file)