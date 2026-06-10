
# Text-to-Mask Vision: Open-Vocabulary Object Detection and Segmentation

## 1. Introduction

This project implements an open-vocabulary image segmentation prototype. The system receives an input image and a natural language prompt, detects the corresponding object with Grounding DINO, and generates a segmentation mask with SAM.

## 2. Problem Statement

Traditional object detectors work with a fixed set of classes. In contrast, open-vocabulary models allow users to describe target objects with text prompts.

## 3. Related Work

* CNN backbones
* DETR-style object detection
* CLIP and vision-language models
* OWL-ViT
* Grounding DINO
* Segment Anything Model

## 4. Methodology

Pipeline:

```text
image + text prompt -> Grounding DINO -> boxes -> SAM -> masks
```

## 5. Implementation

The project includes:

* model wrappers
* visualization utilities
* Streamlit demo
* benchmark utilities

## 6. Experiments

Planned experiments:

* qualitative evaluation on demo images
* latency measurement
* prompt sensitivity analysis
* threshold analysis

## 7. Results

To be filled after experiments.

## 8. Error Analysis

Typical failure cases:

* small objects
* ambiguous prompts
* inaccurate boxes
* mask leakage
* multiple similar objects

## 9. Deployment Demo

The demo is implemented with Streamlit.

## 10. Conclusion and Future Work

The project demonstrates a practical text-to-mask pipeline using pretrained foundation models. Future work may include OWL-ViT comparison, video support, and more systematic evaluation.
