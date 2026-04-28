from __future__ import annotations

from pathlib import Path

import torch

from python.src.eval import evaluate_osr_model
from python.src.eval.osr_evaluator import evaluate_osr_model_with_tsne   # NEW
from python.src.analysis import plot_snr_vs_accuracy                      # NEW


# ── seed → fixed SNR (dB) map ───────────────────────────────────────────────
EVAL_SEED_TO_SNR: dict[int, float] = {
    410:  10,
    118:   8,
    276:   6,
    314:   4,
    152:   2,
    340:   0,
    142:  -2,
    264:  -4,
    336:  -6,
    608:  -8,
    530: -10,
    472: -12,
    214: -14,
}
# ────────────────────────────────────────────────────────────────────────────

# Set to True to generate a t-SNE embedding plot for every SNR point.
# t-SNE is slow — narrow TSNE_SEEDS to just the points you care about.
GENERATE_TSNE = True

# Subset of eval seeds to run t-SNE on.  None = all 13 SNR points.
# Example: [410, 340, 214]  →  only hi (+10 dB), mid (0 dB), lo (-14 dB).
TSNE_SEEDS: list[int] | None = None


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Could not locate project root (no 'artifacts' directory found).")


def main():
    project_root = find_project_root()

    ckpt_seeds        = [38]
    ckpt_n_per_class  = [2500]

    eval_seeds        = list(EVAL_SEED_TO_SNR.keys())
    eval_n_per_class  = [500]
    eval_spec_version = "v2"

    batch_size = 32

    all_results = []

    for ckpt_seed in ckpt_seeds:
        for ckpt_n in ckpt_n_per_class:

            ckpt_tag = f"s{ckpt_seed}_n{ckpt_n}"

            for eval_seed in eval_seeds:
                for eval_n in eval_n_per_class:

                    snr_db    = EVAL_SEED_TO_SNR[eval_seed]
                    snr_label = f"{snr_db:+d} dB"
                    want_tsne = (
                        GENERATE_TSNE
                        and (TSNE_SEEDS is None or eval_seed in TSNE_SEEDS)
                    )

                    print(
                        f"\n\nRunning OSR evaluation "
                        f"| ckpt={ckpt_tag} "
                        f"| eval=seed{eval_seed}_n{eval_n} "
                        f"| SNR={snr_label}"
                        + (" | +t-SNE" if want_tsne else "")
                    )
                    print("=" * 70)

                    if want_tsne:
                        tsne_dir = (
                            project_root
                            / "reports"
                            / "figures"
                            / f"osr_tsne_{ckpt_tag}"
                        )
                        result = evaluate_osr_model_with_tsne(
                            ckpt_seed=ckpt_seed,
                            ckpt_n_per_class=ckpt_n,
                            eval_seed=eval_seed,
                            eval_n_per_class=eval_n,
                            eval_spec_version=eval_spec_version,
                            project_root=project_root,
                            fig_dir=tsne_dir,
                            batch_size=batch_size,
                            snr_label=snr_label,
                        )
                    else:
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

    # ── Text summary ─────────────────────────────────────────────────────── #
    print(f"\n{'=' * 95}")
    print(f"  OSR EVALUATION SUMMARY")
    print(f"{'=' * 95}")
    print(
        f"  {'Ckpt':>12} {'Eval':>12} {'SNR':>6} "
        f"{'KnAcc %':>9} {'AUROC':>8} {'UnkRec %':>10} {'FAR %':>8} {'OSAcc %':>9}"
    )
    print(f"  {'-' * 91}")

    for r in all_results:
        m    = r["metrics"]
        ckpt = f"s{r['checkpoint']['seed']}_n{r['checkpoint']['n_per_class']}"
        evl  = f"s{r['eval_dataset']['seed']}_n{r['eval_dataset']['n_per_class']}"
        snr  = EVAL_SEED_TO_SNR.get(r["eval_dataset"]["seed"], float("nan"))
        print(
            f"  {ckpt:>12} {evl:>12} {snr:>+5}dB "
            f"{100 * m['known_accuracy']:>9.2f} "
            f"{m['auroc']:>8.4f} "
            f"{100 * m['unknown_recall']:>10.2f} "
            f"{100 * m['false_alarm_rate']:>8.2f} "
            f"{100 * m['open_set_accuracy']:>9.2f}"
        )

    print(f"{'=' * 95}\n")

    # ── SNR vs Accuracy plot ─────────────────────────────────────────────── #
    for ckpt_seed in ckpt_seeds:
        for ckpt_n in ckpt_n_per_class:
            subset = [
                r for r in all_results
                if r["checkpoint"]["seed"] == ckpt_seed
                and r["checkpoint"]["n_per_class"] == ckpt_n
            ]
            if not subset:
                continue

            ckpt_tag = f"s{ckpt_seed}_n{ckpt_n}"
            fig_dir  = project_root / "reports" / "figures" / f"osr_snr_sweep_{ckpt_tag}"

            print(f"\nGenerating SNR-accuracy plot → {fig_dir}")
            plot_snr_vs_accuracy(
                results     = subset,
                seed_to_snr = EVAL_SEED_TO_SNR,
                out_dir     = fig_dir,
                ckpt_tag    = ckpt_tag,
            )


if __name__ == "__main__":
    main()