from __future__ import annotations

import torch
from pathlib import Path

from python.src.eval import evaluate_closed_set_model
from python.src.analysis import generate_confusion_outputs, plot_cnn_feature_embedding
from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor
from python.src.utils import (
    create_eval_loader,
    resolve_device,
    FeatureTensorDataset,
    prepare_unique_file
)
from python.src.models import AsymmetricTriNet, SimpleCNN
from python.src.legacy_models import (LiteratureBaseline_ResNet18,
                                      LiteratureBaseline_DenseNet121,
                                      LiteratureBaseline_VGG16)

# Register models locally for diagnostic loading
MODEL_MAP = {
    "asymmetric_trinet": AsymmetricTriNet,
    "simple_cnn": SimpleCNN,
    "vgg_16": LiteratureBaseline_VGG16,
    "densenet121": LiteratureBaseline_DenseNet121,
    "resnet18": LiteratureBaseline_ResNet18,
}


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Could not locate project root (no 'artifacts' directory found).")


def main():
    project_root = find_project_root()
    device = resolve_device("auto")

    # ------------------------------------------------------------------ #
    # User parameters
    # ------------------------------------------------------------------ #
    models = ["vgg_16"]

    # The checkpoint(s) to load
    ckpt_seeds = [55]
    ckpt_n_per_class = [2500]

    # The eval dataset(s) to test against — seeds represent SNR/Impairment levels
    eval_seeds = [410, 118, 276, 314, 152, 340, 142, 264, 336, 608, 530, 472, 214]
    eval_n_per_class = [500]
    eval_spec_version = "v2"

    batch_size = 32

    # ------------------------------------------------------------------ #
    # Sweep
    # ------------------------------------------------------------------ #
    all_results = []

    for model_name in models:
        for ckpt_seed in ckpt_seeds:
            for ckpt_n in ckpt_n_per_class:

                # Load the model once per checkpoint to use for diagnostics
                model_cls = MODEL_MAP[model_name]
                model = model_cls(num_classes=10).to(device)
                ckpt_path = project_root / "artifacts" / "checkpoints" / f"{model_name}_seed{ckpt_seed}_n{ckpt_n}.pt"
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()

                for eval_seed in eval_seeds:
                    for eval_n in eval_n_per_class:

                        print(f"\nEvaluating: {model_name} | Ckpt: s{ckpt_seed} | Eval: s{eval_seed}")

                        # 1. Run Quantitative Metrics (Already saves logs)
                        result = evaluate_closed_set_model(
                            model_name=model_name,
                            ckpt_seed=ckpt_seed,
                            ckpt_n_per_class=ckpt_n,
                            eval_seed=eval_seed,
                            eval_n_per_class=eval_n,
                            eval_spec_version=eval_spec_version,
                            project_root=project_root,
                            batch_size=batch_size,
                        )
                        all_results.append(result)

                        # 2. Prepare Figure Directory
                        fig_base = project_root / "reports" / "figures" / "eval_sweep"
                        fig_folder_name = f"{model_name}_ckpt{ckpt_seed}_eval{eval_seed}"
                        eval_fig_dir = prepare_unique_file(fig_base, fig_folder_name)

                        # 3. Setup DataLoader for Diagnostics (needed for t-SNE/Confusion Matrix)
                        eval_dataset_path = Path("C:/Users/user/Documents/MATLAB/eval_datasets")
                        eval_path = (
                                eval_dataset_path
                                / "impaired"
                                / f"impaired_dataset_{eval_spec_version}_seed{eval_seed}_n{eval_n}_eval.mat"
                        )
                        artifact = load_artifact(str(eval_path), load_params=False)
                        x_stft, x_iq, x_if, y = build_feature_tensor(artifact)
                        dataset = FeatureTensorDataset(x_stft, x_iq, x_if, y)
                        loader = create_eval_loader(dataset, batch_size=batch_size, device=device)

                        # 4. Generate Visual Diagnostics
                        print(f"  Generating diagnostics in: {eval_fig_dir.name}")
                        generate_confusion_outputs(model, loader, device, eval_fig_dir, n_classes=10)
                        plot_cnn_feature_embedding(model, loader, device, eval_fig_dir, n_classes=10)

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary table (UNCHANGED)
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 78}")
    print(f"  {'Model':<22} {'Ckpt':>12} {'Eval':>12} {'Acc %':>8} {'BalAcc %':>10}")
    print(f"  {'-' * 74}")
    for r in all_results:
        m = r["metrics"]
        ckpt = f"s{r['checkpoint']['seed']}_n{r['checkpoint']['n_per_class']}"
        evl = f"s{r['eval_dataset']['seed']}_n{r['eval_dataset']['n_per_class']}"
        print(
            f"  {r['model_name']:<22} {ckpt:>12} {evl:>12} {100 * m['accuracy']:>8.2f} {100 * m['balanced_accuracy']:>10.2f}")


if __name__ == "__main__":
    main()