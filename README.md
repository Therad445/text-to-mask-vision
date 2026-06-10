# Text-to-Mask Vision

Open-vocabulary object detection and segmentation pipeline based on Grounding DINO and Segment Anything Model.

The project implements a simple text-to-mask system:

```text
image + text prompt -> bounding boxes -> segmentation masks
```

The user uploads an image and provides a text prompt such as `bear .`, `dog .`, or `person . car .`. Grounding DINO detects objects matching the prompt, and SAM converts the detected boxes into segmentation masks.

## Project Goal

The goal of this project is to build a practical computer vision demo that combines modern foundation models for open-vocabulary detection and segmentation.

The project focuses on:

* text-guided object detection;
* box-to-mask segmentation;
* prompt-based visual interaction;
* post-processing with non-maximum suppression;
* qualitative error analysis.

## Method

The pipeline consists of four main stages:

1. Load an input image and text prompt.
2. Use Grounding DINO to predict object bounding boxes.
3. Apply non-maximum suppression to reduce duplicate boxes.
4. Use SAM to generate segmentation masks from the selected boxes.

## Models

* Detection: `IDEA-Research/grounding-dino-base`
* Detection interface: Hugging Face Transformers
* Segmentation: Segment Anything Model, ViT-B
* SAM checkpoint: `sam_vit_b_01ec64.pth`

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── src/
│   └── text_to_mask_pipeline.py
├── notebooks/
│   └── 02_colab_hf_grounding_dino_sam_sanity_check.ipynb
├── report/
│   ├── experiments/
│   │   ├── 01_sanity_check.md
│   │   └── 02_streamlit_local_demo.md
│   └── figures/
├── scripts/
│   └── download_sam_weights.sh
└── weights/
```

The `weights/` directory is ignored by Git because model checkpoints are large.

## Results

The first sanity check was performed in Google Colab with a Tesla T4 GPU.

For the prompt:

```text
bear .
```

Grounding DINO detected multiple bear instances. After NMS, the pipeline kept 4 detections and passed them to SAM. SAM successfully generated segmentation masks for the selected boxes.

The local Streamlit demo also works and displays:

* uploaded input image;
* final segmentation result;
* detection labels;
* box confidence scores;
* SAM mask scores;
* bounding box coordinates.

## Demo

The project includes a Streamlit application:

```bash
python -m streamlit run app.py
```

Before running the app, download the SAM checkpoint:

```bash
./scripts/download_sam_weights.sh
```

For CPU-only machines, local inference can be slow. GPU inference via Google Colab is recommended for full experiments.

## Colab Usage

The main working notebook is:

```text
notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb
```

It runs the full pipeline:

```text
image -> Grounding DINO -> NMS -> SAM -> visualization
```

This notebook is the recommended way to reproduce the main experiment.

## Limitations

The model works well for clear objects, but crowded scenes remain challenging.

Observed limitations:

* duplicate boxes before NMS;
* overlapping boxes for close instances;
* partial masks for occluded objects;
* prompt sensitivity;
* slower CPU inference.

The bear example is useful both as a successful demo and as an error-analysis case because several bear instances overlap visually.

## Future Work

Possible improvements:

* add more test images;
* compare different prompts;
* compare Grounding DINO with OWL-ViT;
* add latency measurements;
* add a gallery of success and failure cases;
* improve visualization;
* deploy the Streamlit app on a GPU-enabled environment.

