import torch
from python.src.eval import evaluate_closed_set_model
from python.src.analysis import plot_cnn_feature_embedding
from python.src.utils import resolve_device, create_eval_loader, FeatureTensorDataset
from python.src.dataio import load_artifact
from python.src.preprocessing import build_feature_tensor


def run_snr_sweep(model_name, ckpt_seed, ckpt_n, project_root):
    # Range from -14 to 10 in steps of 2 (or whatever matches your filenames)
    snr_range = range(-14, 12, 2)
    snr_accuracies = []
    device = resolve_device("auto")

    for snr in snr_range:
        # Assuming your MATLAB naming convention follows: impaired_v2_snr{snr}...
        spec_v = "v2"

        print(f"Evaluating SNR: {snr}dB...")

        # 1. Run Standard Evaluation
        result = evaluate_closed_set_model(
            model_name=model_name,
            ckpt_seed=ckpt_seed,
            ckpt_n_per_class=ckpt_n,
            eval_seed=123,  # Fixed eval seed for consistency
            eval_n_per_class=600,
            eval_spec_version=spec_v,
            project_root=project_root
        )
        snr_accuracies.append(result["metrics"]["accuracy"])

        # 2. Generate t-SNE for this specific SNR
        # We need the loader to pass to the diagnostic tool
        eval_path = project_root / "artifacts/datasets/impaired" / f"impaired_dataset_{spec_v}_seed123_n600_eval.mat"
        artifact = load_artifact(str(eval_path), load_params=False)
        x_stft, x_iq, x_if, y = build_feature_tensor(artifact)
        loader = create_eval_loader(FeatureTensorDataset(x_stft, x_iq, x_if, y), batch_size=32, device=device)

        # Load model for t-SNE extraction
        from python.src.models import AsymmetricTriNet  # Or registry
        model = AsymmetricTriNet(num_classes=10).to(device)
        ckpt_path = project_root / f"artifacts/checkpoints/{model_name}_seed{ckpt_seed}_n{ckpt_n}.pt"
        model.load_state_dict(torch.load(ckpt_path))

        fig_dir = project_root / f"reports/figures/snr_sweep/snr_{snr}"
        plot_cnn_feature_embedding(model, loader, device, fig_dir, n_classes=10)

    return list(snr_range), snr_accuracies