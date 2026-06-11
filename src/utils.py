from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_float_list(values: Iterable[float], digits: int = 4) -> str:
    """Format a list of floats for compact CSV/markdown reporting."""
    return ", ".join(f"{value:.{digits}f}" for value in values)


def mean_or_zero(values: Iterable[float]) -> float:
    """Return the mean value, or 0.0 for an empty iterable."""
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def normalize_prompt(prompt: str) -> str:
    """
    Normalize a Grounding DINO prompt.

    Grounding DINO works better when object names are separated by periods:
    'bus . person .' instead of 'bus person'.
    """
    prompt = prompt.strip()
    if not prompt:
        return prompt
    if not prompt.endswith("."):
        prompt += " ."
    return prompt
