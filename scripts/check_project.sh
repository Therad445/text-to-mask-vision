#!/usr/bin/env bash
set -euo pipefail

echo "Checking project structure..."

required_paths=(
"README.md"
"requirements.txt"
"app.py"
"src/pipeline.py"
"src/grounding_dino_detector.py"
"src/sam_segmenter.py"
"src/visualization.py"
"data/demo_images"
"outputs/predictions"
"report/final_report.md"
"presentation/slides.md"
)

for path in "${required_paths[@]}"; do
if [ ! -e "$path" ]; then
echo "Missing: $path"
exit 1
fi
done

echo "Project check passed."
