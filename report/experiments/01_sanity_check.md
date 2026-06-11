# Experiment 01 — HF Grounding DINO + SAM Sanity Check

## Goal

The goal of this experiment was to run the first working text-to-mask pipeline:

```text
image + text prompt -> bounding boxes -> segmentation masks
```

This sanity check verifies that the core project idea is technically feasible before moving to a reusable Python pipeline and Streamlit demo.

## Setup

* Runtime: Google Colab
* GPU: Tesla T4
* Detection model: `IDEA-Research/grounding-dino-base`
* Detection interface: Hugging Face Transformers
* Segmentation model: Segment Anything Model, ViT-B
* SAM checkpoint: `sam_vit_b_01ec64.pth`
* Input image: demo image with several bears
* Text prompt: `bear .`

## Parameters

```text
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 4
```

## Pipeline

The experiment used the following pipeline:

```text
PIL image
    -> Grounding DINO processor
    -> Grounding DINO model
    -> candidate bounding boxes
    -> non-maximum suppression
    -> SAM predictor
    -> segmentation masks
    -> final visualization
```

## Detection Results

Grounding DINO initially returned 8 candidate boxes for the prompt `bear .`.

After applying non-maximum suppression, 4 boxes were kept.

The final retained boxes had confidence scores approximately in the range:

```text
0.42 - 0.46
```

## Segmentation Results

SAM successfully generated masks for the retained boxes.

The output mask tensor shape was:

```text
[4, 1, 1450, 2068]
```

The SAM mask scores were:

```text
0.9759
0.9608
0.9283
0.9399
```

The final visualization was saved as:

```text
outputs/predictions/sanity_text_to_mask_hf.png
```

A copy of the same figure is also used in the report figures directory:

```text
report/figures/sanity_text_to_mask_hf.png
```

## Observations

The experiment confirms that the project core is working: a text prompt can be used to detect objects, and the resulting boxes can be passed to SAM to obtain segmentation masks.

Grounding DINO was able to find multiple bear instances in a crowded scene. SAM successfully converted the selected bounding boxes into object masks.

The result also shows an important limitation: because several bears overlap visually, the detector can produce duplicate or partially overlapping boxes. Non-maximum suppression improves readability, but it may also suppress valid detections in crowded scenes.

## Error Analysis Notes

This example is useful not only as a success case, but also as an early failure-analysis case.

Observed issues:

* Several detected boxes overlap.
* One bear/cub is segmented only partially.
* Close object instances are hard to separate.
* NMS reduces duplicate boxes but can remove valid nearby detections.
* The final quality depends strongly on the quality of the bounding boxes.

Possible improvements:

* Tune `BOX_THRESHOLD` and `TEXT_THRESHOLD`.
* Tune `NMS_IOU_THRESHOLD`.
* Keep more detections for crowded scenes.
* Test alternative prompts such as `bear cub .` or `adult bear . bear cub .`
* Use this image as a limitation example and use cleaner images for the main demo.

## Conclusion

The first sanity check is successful. The project has a working foundation:

```text
text prompt -> object detection -> bounding boxes -> segmentation masks
```

The next step is to move this logic from the Colab notebook into reusable project modules:

```text
src/grounding_dino_detector.py
src/sam_segmenter.py
src/pipeline.py
src/visualization.py
```

After that, the pipeline can be connected to the Streamlit application.

Note: this file documents an initial sanity/demo run. The final tuned crowded-bears setting is reported in `03_gallery_results.csv`, `04_evaluation_summary.md`, the final report, and the presentation.
