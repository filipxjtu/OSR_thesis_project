from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.floating]


class DatasetView(Protocol):

    """ Abstract view used by the validator """

    @property
    def name(self) -> str: ...

    # Time-domain signals: shape (N_samples, N_time)
    def x_time(self) -> ArrayF: ...

    # Labels: shape (N_samples,)
    def y(self) -> npt.NDArray[np.integer]: ...

    # Optional: feature tensor e.g., (N_samples, C, F, T) or (N_samples, F, T)
    def x_feat(self) -> Any | None: ...

    # Metadata dict (must include spec_version, mode, etc.)
    def meta(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DatasetBundle:
    clean: DatasetView
    impaired_train: DatasetView
    impaired_eval: DatasetView

    def datasets(self):
        return self.clean, self.impaired_train, self.impaired_eval