from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def parse_scores(value: str) -> list[float]:
    """Parse a comma-separated list of floating-point scores."""
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_gallery(csv_path: Path) -> list[dict[str, str]]:
    """Read gallery CSV results and compute compact per-case summary rows."""
    rows: list[dict[str, str]] = []

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            box_scores = parse_scores(row.get("box_scores", ""))
            mask_scores = parse_scores(row.get("mask_scores", ""))

            rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "prompt": row.get("prompt", ""),
                    "raw_detections": row.get("raw_detections", ""),
                    "detections_after_nms": row.get("detections_after_nms", ""),
                    "mean_box_score": f"{mean(box_scores):.3f}",
                    "mean_mask_score": f"{mean(mask_scores):.3f}",
                }
            )

    return rows


def print_markdown_table(rows: list[dict[str, str]]) -> None:
    """Print a markdown table with gallery summary metrics."""
    print("| Case | Prompt | Raw detections | Final detections | Mean box score | Mean mask score |")
    print("|---|---|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| `{row['case_id']}` | `{row['prompt']}` | "
            f"{row['raw_detections']} | {row['detections_after_nms']} | "
            f"{row['mean_box_score']} | {row['mean_mask_score']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gallery experiment metrics.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("report/experiments/03_gallery_results.csv"),
        help="Path to gallery CSV results.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    rows = summarize_gallery(args.csv)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
