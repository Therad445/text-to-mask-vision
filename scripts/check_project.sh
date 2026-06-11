#!/usr/bin/env bash
set -euo pipefail

echo "Checking project structure..."

required_paths=(
  "README.md"
  "requirements.txt"
  "app.py"
  "src/text_to_mask_pipeline.py"
  "notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb"
  "notebooks/03_colab_batch_gallery.ipynb"
  "report/final_report.md"
  "report/final_report.pdf"
  "report/experiments/01_sanity_check.md"
  "report/experiments/02_streamlit_local_demo.md"
  "report/experiments/03_gallery_results.csv"
  "report/experiments/03_gallery_summary.md"
  "report/experiments/04_evaluation_summary.md"
  "report/figures/streamlit_demo_result.png"
  "report/figures/streamlit_detections_table.png"
  "report/figures/sanity_text_to_mask_hf.png"
  "report/figures/gallery/01_crowded_bears.png"
  "report/figures/gallery/03_bus_person_multi_object.png"
  "report/figures/gallery/05_weak_prompt_case.png"
  "presentation/slides.md"
  "presentation/text_to_mask_vision_final_presentation.pdf"
  "presentation/text_to_mask_vision_final_presentation.pptx"
  "presentation/speaker_notes_ru.md"
  "scripts/download_sam_weights.sh"
)

for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "Missing: $path"
    exit 1
  fi
done

echo "Checking that large/local files are not tracked..."

if git ls-files | grep -E '(^\.venv/|^weights/|\.pth$|\.zip$|__pycache__|\.pyc$)'; then
  echo "Unexpected local/large files are tracked."
  exit 1
fi

echo "Checking for unfinished placeholder code..."

if grep -R --exclude='*.pdf' --exclude='check_project.sh' "NotImplementedError" -n src scripts notebooks README.md report presentation; then
  echo "Found unfinished placeholder code."
  exit 1
fi

echo "Project check passed."
