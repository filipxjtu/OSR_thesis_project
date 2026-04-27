from __future__ import annotations

from pathlib import Path

import torch

from python.src.eval import evaluate_osr_model


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Could not locate project root (no 'artifacts' directory found).")


def main():
    project_root = find_project_root()

    ckpt_seeds        = [55]
    ckpt_n_per_class  = [2500]

    eval_seeds        = [400, 102, 276, 312, 154, 346, 145, 265]
    eval_n_per_class  = [600]
    eval_spec_version = "v2"

    batch_size = 32

    all_results = []

    for ckpt_seed in ckpt_seeds:
        for ckpt_n in ckpt_n_per_class:
            for eval_seed in eval_seeds:
                for eval_n in eval_n_per_class:

                    print(
                        f"\n\nRunning OSR evaluation "
                        f"| ckpt=seed{ckpt_seed}_n{ckpt_n} "
                        f"| eval=seed{eval_seed}_n{eval_n}"
                    )
                    print("=" * 70)

                    result = evaluate_osr_model(
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

    print(f"\n{'=' * 90}")
    print(f"  OSR EVALUATION SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"  {'Ckpt':>12} {'Eval':>12} "
        f"{'KnAcc %':>9} {'AUROC':>8} {'UnkRec %':>10} {'FAR %':>8} {'OSAcc %':>9}"
    )
    print(f"  {'-' * 86}")

    for r in all_results:
        m    = r["metrics"]
        ckpt = f"s{r['checkpoint']['seed']}_n{r['checkpoint']['n_per_class']}"
        evl  = f"s{r['eval_dataset']['seed']}_n{r['eval_dataset']['n_per_class']}"
        print(
            f"  {ckpt:>12} {evl:>12} "
            f"{100 * m['known_accuracy']:>9.2f} "
            f"{m['auroc']:>8.4f} "
            f"{100 * m['unknown_recall']:>10.2f} "
            f"{100 * m['false_alarm_rate']:>8.2f} "
            f"{100 * m['open_set_accuracy']:>9.2f}"
        )

    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()