from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from ..dataio.dataset_artifact import DatasetArtifact


# Dataset role typing

DatasetRole = Literal[
    "clean",
    "impaired_train",
    "impaired_eval",
    "unknown",
    "clean_unk",
]


# Single dataset wrapper

@dataclass(frozen=True)
class Dataset:
    artifact: DatasetArtifact
    role: DatasetRole

    @property
    def X(self):
        return self.artifact.X

    @property
    def y(self):
        return self.artifact.y

    @property
    def meta(self):
        return self.artifact.meta

    @property
    def name(self) -> str:
        return self.role


# Bundle definition

@dataclass(frozen=True)
class DatasetBundle:
    """ Represents the full validation input space. """

    clean: Dataset
    impaired_train: Dataset
    impaired_eval: Dataset

    unknown: Optional[Dataset] = None
    clean_unk: Optional[Dataset] = None


    # Convenience accessors
    def known_datasets(self) -> list[Dataset]:
        return [
            self.clean,
            self.impaired_train,
            self.impaired_eval,
        ]

    def unknown_datasets(self) -> list[Dataset]:
        out = []
        if self.unknown is not None:
            out.append(self.unknown)
        if self.clean_unk is not None:
            out.append(self.clean_unk)
        return out

    def all_datasets(self) -> list[Dataset]:
        return self.known_datasets() + self.unknown_datasets()


    # Flags
    @property
    def has_unknown(self) -> bool:
        return self.unknown is not None

    @property
    def has_unknown_clean(self) -> bool:
        return self.clean_unk is not None