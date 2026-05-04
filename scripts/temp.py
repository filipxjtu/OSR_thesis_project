import torch
from pathlib import Path
from python.src.models import OsrSAF_TriNet



def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for p in current.parents:
        if (p / "artifacts").exists():
            return p
    raise RuntimeError("Could not locate project root (no 'artifacts' directory found).")



# 1. Initialize the model (using your standard 10 classes)
model = OsrSAF_TriNet(num_classes=10)

proot = find_project_root()

# 2. Load your trained checkpoint
# (Change the seed or n_per_class in the filename if you are testing a different run)
ckpt_path = proot / f"artifacts/checkpoints/osr_saf_trinet_seed216_n2500.pt"
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

# 3. Extract and print the specific Hamming stats
stats = model._hamming_codebook.convergence_stats()

print("=== Hamming Spread per Class ===")
# This shows how tightly clustered the binary survival masks are for each class
print(stats['spread_per_class_hamming'])

print("\n=== Firing Rate per Class ===")
# This shows the average number of FusedDRSN channels that survive shrinkage per class
print(stats['firing_rate_per_class'])


proto = (model._hamming_codebook.prototypes_soft >= 0.5).float()  # (C, k, D)
# average prototype per class (collapse k)
mean_proto = proto.mean(dim=1)                                    # (C, D)
# pairwise hamming distances between class prototypes
diff = mean_proto.unsqueeze(0) - mean_proto.unsqueeze(1)          # (C, C, D)
between = diff.abs().mean(dim=-1)                                 # (C, C)
# zero out the diagonal for readability
between.fill_diagonal_(0)
print("Between-class Hamming distance matrix:")
print(between.round(decimals=3))
print(f"\nMean off-diagonal: {between.sum() / (between.numel() - between.size(0)):.4f}")
print(f"Max off-diagonal:  {between.max():.4f}")
print(f"Min off-diagonal:  {between[between > 0].min():.4f}")