"""Render explanation plots from a predict run.

Usage: uv run --group plots python plots.py <run_dir | explanations.csv> [-o out_dir]

Reads the *.explanations*.csv sidecar written by predict.py and renders every
chart as a matplotlib PNG (png/) and a Highcharts config JSON plus a
self-contained HTML page (highcharts/).
"""

import argparse
from pathlib import Path

from chap_explain import explain, plotting


def main(path: str, out_dir: str | None) -> None:
    source = Path(path)
    explanations = plotting.find_explanations_file(source)
    tidy = plotting.load_explanations(explanations)
    importance = explain.global_importance(tidy[tidy.view.isin(["lagged", "base"])])
    target = Path(out_dir) if out_dir else explanations.parent / "plots"
    written = plotting.render_all(tidy, importance, target)
    print(f"Read {explanations}")
    print(f"Wrote {len(written)} files under {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Run directory or explanations CSV")
    parser.add_argument("-o", "--out-dir", default=None)
    args = parser.parse_args()
    main(args.path, args.out_dir)
