from __future__ import annotations


class ArtifactLoadError(Exception):
    """Base class for any artifact loading/validation failure."""

class ContractViolationError(ArtifactLoadError):
    """Raised when the artifact violates the interface contract."""

class RootNotFoundError(ContractViolationError):
    """Expected root variable (dataset or impaired_data) not found."""

class MultipleRootError(ContractViolationError):
    """Multiple competing roots found (ambiguous artifact)."""

class MissingFieldError(ContractViolationError):
    """Required field missing from artifact."""

class ShapeMismatchError(ContractViolationError):
    """Array shape mismatch vs contract."""

class DtypeMismatchError(ContractViolationError):
    """Array dtype mismatch vs contract."""

class MetadataError(ContractViolationError):
    """Metadata missing, inconsistent, or invalid."""

class AlignmentError(ContractViolationError):
    """Sample alignment mismatch across X/y/params/imp_params."""

class NumericDomainError(ContractViolationError):
    """NaN/Inf or other forbidden numeric values detected."""

class HashMismatchError(ContractViolationError):
    """Recomputed hash does not match meta.artifact_hash."""



