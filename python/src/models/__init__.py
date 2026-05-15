from .simple_cnn import SimpleCNN
from .osr_saf_trinet import OsrSAF_TriNet
from .asymmetric_trinet import AsymmetricTriNet
from .ablation_trinet import AblationTriNet
from .ablation_osr_saf_trinet import AblationOsrSAF_TriNet
from .openmax_trinet import  OpenMaxTriNet

__all__ = [
    "SimpleCNN",
    "OsrSAF_TriNet",
    "AsymmetricTriNet",
    "AblationTriNet",
    "AblationOsrSAF_TriNet",
    "OpenMaxTriNet",
]