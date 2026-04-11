# python/src/utils/file_saver.py

from pathlib import Path


def get_unique_path(base_path: Path) -> Path:
    """
    Returns a collision-safe path.
    If base_path exists, appends (2), (3), ... before suffix.

    Example:
        gold.pt → gold(2).pt → gold(3).pt
    """
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    i = 2
    while True:
        candidate = parent / f"{stem}({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def prepare_unique_file(base_dir: Path, filename: str) -> Path:
    """
    Ensures directory exists and returns a collision-safe file path.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / filename
    return get_unique_path(base_path)