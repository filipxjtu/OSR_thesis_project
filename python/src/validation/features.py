from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import FailedCheck
from .stats import effect_size_delta


@dataclass(frozen=True)
class FeatureThresholds:
    # Feature finite + non-collapse
    feat_std_min: float = 1e-6
    feat_energy_min: float = 1e-8

    # Separation in feature space
    min_effect_size_feat_train: float = 0.20
    min_effect_size_feat_eval: float = 0.20
    min_effect_size_feat_train_vs_eval: float = 0.10


def _require(cond: bool, check_id: str, message: str, details: dict[str, Any]) -> list[FailedCheck]:
    return [] if cond else [FailedCheck(check_id=check_id, message=message, details=details)]


def _as_feat_array(x_feat: Any) -> np.ndarray:
    """
    Accepts np arrays or torch tensors; returns float64 numpy.
    Expected shape: (N, ...) any rank >= 2.
    """
    if x_feat is None:
        raise ValueError("x_feat is None")
    try:
        import torch  # optional
        if isinstance(x_feat, torch.Tensor):
            x_feat = x_feat.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x_feat, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"Feature tensor must have ndim>=2, got {x.ndim}")
    return x


def feature_checks(bundle, th: FeatureThresholds) -> tuple[list[FailedCheck], dict[str, Any]]:
    """
    Runs only if ALL datasets provide x_feat != None.
    """
    feats = {}
    for ds in (bundle.clean, bundle.impaired_train, bundle.impaired_eval):
        xf = ds.x_feat()
        if xf is None:
            return [], {"skipped": True, "reason": f"{ds.name} has no x_feat()"}
        feats[ds.name] = _as_feat_array(xf)

    fails: list[FailedCheck] = []
    metrics: dict[str, Any] = {"skipped": False}

    # Finite checks + collapse checks
    for name, x in feats.items():
        bad = ~np.isfinite(x)
        fails += _require(
            not bool(np.any(bad)),
            "F001.feat_no_nan_inf",
            f"{name}: found NaN/Inf in feature tensor",
            {"dataset": name, "bad_count": int(np.sum(bad))},
        )

        std = float(np.std(x))
        fails += _require(
            std >= th.feat_std_min,
            "F002.feat_not_collapsed",
            f"{name}: feature std too small (collapse)",
            {"dataset": name, "std": std, "threshold": th.feat_std_min},
        )

        # per-sample energy sanity: mean(sum(x^2)) must not vanish
        x_flat = x.reshape(x.shape[0], -1)
        energy = np.mean(np.sum(x_flat * x_flat, axis=1))
        energy = float(energy)
        fails += _require(
            energy >= th.feat_energy_min,
            "F003.feat_energy_nonzero",
            f"{name}: feature energy too small (degenerate features)",
            {"dataset": name, "energy": energy, "threshold": th.feat_energy_min},
        )

        metrics[name] = {"std": std, "mean_energy": energy, "shape": list(x.shape)}

    # Separation checks (feature space)
    c = feats[bundle.clean.name].reshape(feats[bundle.clean.name].shape[0], -1)
    tr = feats[bundle.impaired_train.name].reshape(feats[bundle.impaired_train.name].shape[0], -1)
    ev = feats[bundle.impaired_eval.name].reshape(feats[bundle.impaired_eval.name].shape[0], -1)

    d_tr = effect_size_delta(c, tr)
    d_ev = effect_size_delta(c, ev)
    d_te = effect_size_delta(tr, ev)

    metrics["effect_sizes_feat"] = {
        "clean_vs_train": float(d_tr),
        "clean_vs_eval": float(d_ev),
        "train_vs_eval": float(d_te),
    }

    fails += _require(
        d_tr >= th.min_effect_size_feat_train,
        "F010.feat_sep_clean_vs_train",
        "Feature-space separation too small: clean vs impaired_train",
        {"effect_size": float(d_tr), "threshold": th.min_effect_size_feat_train},
    )
    fails += _require(
        d_ev >= th.min_effect_size_feat_eval,
        "F011.feat_sep_clean_vs_eval",
        "Feature-space separation too small: clean vs impaired_eval",
        {"effect_size": float(d_ev), "threshold": th.min_effect_size_feat_eval},
    )
    fails += _require(
        d_te >= th.min_effect_size_feat_train_vs_eval,
        "F012.feat_sep_train_vs_eval",
        "Feature-space separation too small: impaired_train vs impaired_eval",
        {"effect_size": float(d_te), "threshold": th.min_effect_size_feat_train_vs_eval},
    )

    return fails, metrics