# Experiment 02 — Local Streamlit Demo

## Goal

The goal of this experiment was to verify that the text-to-mask pipeline can be used through a local interactive web interface.

## Setup

- Environment: WSL / local laptop
- Interface: Streamlit
- Detection model: Grounding DINO via Hugging Face Transformers
- Segmentation model: SAM ViT-B
- Prompt: `bear .`
- Box threshold: 0.25
- Text threshold: 0.25
- NMS IoU threshold: 0.45
- Max detections: 4

## Result

The Streamlit application successfully started locally and accepted an uploaded image.

The pipeline produced:

- 4 detected objects;
- 4 segmentation masks;
- detection confidence scores around 0.36–0.44;
- mask scores around 0.96–0.98.

The result was displayed in the web interface together with a detections table containing labels, box scores, mask scores, and bounding box coordinates.

## Observation

The local demo confirms that the project is not only a notebook experiment, but also an interactive application.

The model is computationally heavy for CPU-only inference, so Google Colab GPU remains the recommended environment for full experiments. However, the local Streamlit demo is functional and suitable for demonstrating the end-to-end pipeline.

## Limitation

The bear image is a crowded scene with several overlapping instances. As a result, some masks overlap and one instance is only partially segmented. This case is useful for error analysis.

Note: this file documents an initial sanity/demo run. The final tuned crowded-bears setting is reported in `03_gallery_results.csv`, `04_evaluation_summary.md`, the final report, and the presentation.
