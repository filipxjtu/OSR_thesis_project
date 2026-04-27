from __future__ import annotations

from pathlib import Path

import torch

from python.src.eval import evaluate_closed_set_model


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Could not locate project root (no 'artifacts' directory found).")


def main():
    project_root = find_project_root()

    # ------------------------------------------------------------------ #
    # User parameters
    # ------------------------------------------------------------------ #
    models = ["simple_cnn"]

    # The checkpoint(s) to load — what the model was trained on
    ckpt_seeds        = [216]
    ckpt_n_per_class  = [2500]

    # The eval dataset(s) to test against — independent of the checkpoint
    eval_seeds        = [400, 102, 276, 312, 154, 346, 145, 265]
    eval_n_per_class  = [600]
    eval_spec_version = "v2"

    batch_size = 64

    # ------------------------------------------------------------------ #
    # Sweep
    # ------------------------------------------------------------------ #
    all_results = []

    for model_name in models:
        for ckpt_seed in ckpt_seeds:
            for ckpt_n in ckpt_n_per_class:
                for eval_seed in eval_seeds:
                    for eval_n in eval_n_per_class:

                        print(
                            f"\n\nRunning evaluation | model={model_name} "
                            f"| ckpt=seed{ckpt_seed}_n{ckpt_n} "
                            f"| eval=seed{eval_seed}_n{eval_n}"
                        )
                        print("=" * 70)

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

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            torch.mps.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 78}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 78}")
    print(
        f"  {'Model':<22} {'Ckpt':>12} {'Eval':>12} "
        f"{'Acc %':>8} {'BalAcc %':>10} {'F1-macro':>10}"
    )
    print(f"  {'-' * 74}")

    for r in all_results:
        m    = r["metrics"]
        ckpt = f"s{r['checkpoint']['seed']}_n{r['checkpoint']['n_per_class']}"
        evl  = f"s{r['eval_dataset']['seed']}_n{r['eval_dataset']['n_per_class']}"
        print(
            f"  {r['model_name']:<22} {ckpt:>12} {evl:>12} "
            f"{100 * m['accuracy']:>8.2f} "
            f"{100 * m['balanced_accuracy']:>10.2f} "
            f"{m['f1_macro']:>10.4f}"
        )

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()