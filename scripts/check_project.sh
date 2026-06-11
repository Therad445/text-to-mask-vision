#!/usr/bin/env bash
set -euo pipefail

echo "Checking project structure..."

required_paths=(
  "README.md"
  "requirements.txt"
  "app.py"
  "configs/prompts.yaml"
  "src/text_to_mask_pipeline.py"
  "src/utils.py"
  "src/benchmark.py"
  "notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb"
  "notebooks/03_colab_batch_gallery.ipynb"
  "report/final_report.md"
  "report/final_report.pdf"
  "report/experiments/01_sanity_check.md"
  "report/experiments/02_streamlit_local_demo.md"
  "report/experiments/03_gallery_results.csv"
  "report/experiments/03_gallery_summary.md"
  "report/experiments/04_evaluation_summary.md"
  "report/figures/sanity_text_to_mask_hf.png"
  "report/figures/streamlit_demo_result.png"
  "report/figures/streamlit_detections_table.png"
  "report/figures/gallery/01_crowded_bears.png"
  "report/figures/gallery/02_bear_cub_prompt.png"
  "report/figures/gallery/03_bus_person_multi_object.png"
  "report/figures/gallery/05_weak_prompt_case.png"
  "presentation/slides.md"
  "presentation/text_to_mask_vision_final_presentation.pdf"
  "presentation/text_to_mask_vision_final_presentation.pptx"
  "presentation/speaker_notes_ru.md"
  "scripts/download_sam_weights.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path"
    exit 1
  fi
done

echo "Checking that large/local files are not tracked..."
if git ls-files | grep -E '(^\.venv/|^weights/|\.pth$|\.zip$|__pycache__|\.pyc$)' >/dev/null; then
  echo "Large or local-only files are tracked:"
  git ls-files | grep -E '(^\.venv/|^weights/|\.pth$|\.zip$|__pycache__|\.pyc$)'
  exit 1
fi

echo "Checking Python syntax..."
python3 -m py_compile app.py src/*.py scripts/*.py

echo "Checking config files..."
python3 - <<'PY'
from pathlib import Path

p = Path("configs/prompts.yaml")
text = p.read_text(encoding="utf-8")

bad_patterns = [
    "cat >",
    "<<'EOF'",
    "<<EOF",
    "\nEOF\n",
    "from **future**",
    "```",
]

for pattern in bad_patterns:
    if pattern in text:
        raise SystemExit(f"Bad artifact in {p}: {pattern!r}")

print("Config looks clean.")
PY

echo "Checking source files for accidental copied markdown artifacts..."
python3 - <<'PY'
from pathlib import Path

paths = list(Path("src").glob("*.py")) + [Path("app.py")]

bad_patterns = [
    "from **future**",
    "```",
    "cat >",
    "<<'EOF'",
    "<<EOF",
]

for path in paths:
    text = path.read_text(encoding="utf-8")
    for pattern in bad_patterns:
        if pattern in text:
            raise SystemExit(f"Bad artifact in {path}: {pattern!r}")

print("Source files look clean.")
PY

echo "Checking for unfinished placeholder code..."
if grep -R -n "NotImplementedError" app.py src scripts --exclude='check_project.sh'; then
  echo "Found unfinished placeholder code."
  exit 1
fi

echo "Project check passed."
