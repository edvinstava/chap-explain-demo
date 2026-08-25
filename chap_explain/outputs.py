"""Output file naming for explanation sidecar files.

CHAP's counterfactual command (`chap causal`) calls predict twice in the same
run directory (original scenario first, counterfactual second), so sidecar
files must never overwrite an earlier call's output. Paths are derived from
the predictions out_file and suffixed _2, _3, ... on collision.
"""

from __future__ import annotations

from pathlib import Path

SIDECAR_KINDS = ("explanations", "global_importance")


def sidecar_paths(out_file: str) -> dict[str, Path]:
    """Collision-free sidecar paths for one predict call, sharing one suffix."""
    stem = Path(out_file).with_suffix("")
    suffix = 0
    while True:
        suffix += 1
        tag = "" if suffix == 1 else f"_{suffix}"
        paths = {kind: Path(f"{stem}.{kind}{tag}.csv") for kind in SIDECAR_KINDS}
        if not any(path.exists() for path in paths.values()):
            return paths
