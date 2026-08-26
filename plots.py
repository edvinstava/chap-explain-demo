"""Render explanation plots from a predict run.

Usage: uv run --group plots python plots.py <run_dir | explanations.csv> [-o out_dir]
           [--locations A,B] [--periods 2023-10,2023-11] [--charts local,over_time]

Reads the *.explanations*.csv sidecar written by predict.py and renders each
chart as a matplotlib PNG (png/) and a Highcharts config JSON plus a
self-contained HTML page (highcharts/).

By default only the lean chart set is rendered: global importance, plus one
local "why this prediction" chart (at the predicted peak) and one over-time
drivers chart per location. Intervention charts are opt-in via
--charts intervention.
"""

import argparse
from pathlib import Path

from chap_explain import explain, plotting


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(
    path: str,
    out_dir: str | None,
    locations: str | None = None,
    periods: str | None = None,
    charts: str | None = None,
) -> None:
    source = Path(path)
    explanations = plotting.find_explanations_file(source)
    tidy = plotting.load_explanations(explanations)
    tidy = plotting.filter_explanations(tidy, _split(locations), _split(periods))
    importance = explain.global_importance(tidy[tidy.view.isin(["lagged", "base"])])
    target = Path(out_dir) if out_dir else explanations.parent / "plots"
    written = plotting.render_all(
        tidy, importance, target, charts=_split(charts) or plotting.DEFAULT_CHARTS
    )
    print(f"Read {explanations}")
    print(f"Wrote {len(written)} files under {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Run directory or explanations CSV")
    parser.add_argument("-o", "--out-dir", default=None)
    parser.add_argument(
        "--locations", default=None, help="Comma-separated locations to plot (default: all)"
    )
    parser.add_argument(
        "--periods", default=None, help="Comma-separated time periods to plot (default: all)"
    )
    parser.add_argument(
        "--charts",
        default=None,
        help=(
            "Comma-separated chart families from "
            f"{','.join(plotting.CHART_FAMILIES)} "
            f"(default: {','.join(plotting.DEFAULT_CHARTS)})"
        ),
    )
    args = parser.parse_args()
    main(args.path, args.out_dir, args.locations, args.periods, args.charts)
