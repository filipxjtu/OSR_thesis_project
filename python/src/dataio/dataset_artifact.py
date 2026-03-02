
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DatasetArtifact:
    # clean: X = X_clean, impaired: X = X_imp

    X: npt.NDArray[np.floating]
    y: npt.NDArray[np.integer]
    params: Any
    imp_params: Optional[Any]
    meta: dict[str, Any]
    root: str  # "dataset" or "impaired_data"

