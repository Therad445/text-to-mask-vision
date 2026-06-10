# Text-to-Mask Vision

## Open-Vocabulary Object Detection and Segmentation with Grounding DINO and SAM

**Author:** Radmir Islamov  
**Course:** Computer Vision Final Project  
**Repository:** `github.com/Therad445/text-to-mask-vision`

---

# 1. Project Idea

The project implements a practical text-to-mask computer vision pipeline:

```text
image + text prompt -> bounding boxes -> segmentation masks
```

The user uploads an image and writes a prompt such as:

```text
bear .
person .
bus . person .
```

The system detects objects matching the prompt and converts detections into segmentation masks.

---

# 2. Problem Statement

Classical object detectors usually work with a fixed set of classes.

This project explores a more flexible setting:

- the user describes the target object in natural language;
- the detector finds matching regions;
- the segmentation model produces pixel-level masks;
- no additional training is required.

Potential use cases:

- interactive image annotation;
- visual search;
- dataset labeling;
- rapid prototyping of CV pipelines.

---

# 3. Main Models

The project combines two pretrained vision foundation models.

## Grounding DINO

Used for open-vocabulary object detection.

Input:

```text
image + text prompt
```

Output:

```text
boxes + scores + labels
```

## Segment Anything Model

Used for box-guided segmentation.

Input:

```text
image + selected boxes
```

Output:

```text
segmentation masks + mask scores
```

---

# 4. Pipeline Architecture

```text
Input image
    |
    v
Text prompt normalization
    |
    v
Grounding DINO detection
    |
    v
Non-Maximum Suppression
    |
    v
SAM box-to-mask segmentation
    |
    v
Visualization + detections table
```

Default parameters:

- box threshold: `0.25`;
- text threshold: `0.25`;
- NMS IoU threshold: `0.45`;
- max detections: configurable.

---

# 5. Implementation

The project contains both notebook experiments and an interactive demo.

Main files:

```text
src/text_to_mask_pipeline.py
app.py
notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb
notebooks/03_colab_batch_gallery.ipynb
report/experiments/
report/figures/
```

The reusable pipeline module implements:

- model loading;
- detection;
- NMS filtering;
- SAM segmentation;
- end-to-end inference.

---

# 6. Streamlit Demo

The project includes a local Streamlit interface.

The demo supports:

- image upload;
- text prompt input;
- box/text threshold controls;
- NMS IoU threshold control;
- maximum detections control;
- result visualization;
- detections table with scores and coordinates.

Example command:

```bash
python -m streamlit run app.py
```

![Streamlit demo](../report/figures/streamlit_demo_result.png)

---

# 7. Experiments

Three experiment blocks were prepared.

## Experiment 01 — Sanity Check

Goal: verify that the full pipeline works end-to-end.

Prompt:

```text
bear .
```

Result:

```text
image -> Grounding DINO -> NMS -> SAM -> visualization
```

## Experiment 02 — Local Demo

Goal: verify that the pipeline works through Streamlit.

## Experiment 03 — Batch Gallery

Goal: test several success and failure cases.

---

# 8. Batch Gallery

The batch gallery was run in Google Colab with a Tesla T4 GPU.

Test cases:

| Case | Prompt | Purpose |
|---|---|---|
| Crowded bears | `bear .` | crowded scene / error analysis |
| Bear cub prompt | `bear cub .` | prompt sensitivity |
| Bus and people | `bus . person .` | multi-object prompt |
| Clear person | `person .` | simple success case |
| Weak prompt | `small animal .` | mismatched prompt |

Outputs:

- 5 result images;
- CSV table with scores;
- markdown summary.

---

# 9. Qualitative Results

The strongest results are obtained when:

- the object is clearly visible;
- the prompt is simple and specific;
- objects are not heavily occluded;
- the image contains common object categories.

Example result:

![Batch gallery example](../report/figures/gallery/03_bus_person_multi_object.png)

The system successfully demonstrates open-vocabulary detection followed by segmentation.

---

# 10. Error Analysis

Observed limitations:

- Grounding DINO may produce overlapping boxes;
- crowded scenes are difficult;
- NMS can remove valid nearby objects;
- inaccurate boxes can lead to partial SAM masks;
- prompt wording affects results;
- CPU inference is slow.

The bear example is useful because it is both a working demo and a failure-analysis case.

![Crowded bears](../report/figures/gallery/01_crowded_bears.png)

---

# 11. Individual Contribution

This is an individual practical project.

Implemented and prepared:

- repository structure;
- reusable text-to-mask pipeline;
- Colab sanity notebook;
- batch gallery notebook;
- Streamlit demo;
- experiment logs;
- visual results;
- final report draft;
- README and usage instructions.

Large pretrained checkpoints are not stored in Git and are downloaded separately.

---

# 12. Conclusion

The project implements a working text-to-mask computer vision system.

Final result:

```text
image + text prompt -> open-vocabulary boxes -> masks -> interactive demo
```

The project demonstrates how modern pretrained foundation models can be combined into a practical CV application without training a new model.

Future work:

- more test images;
- latency measurements;
- comparison with OWL-ViT;
- mask export as COCO/PNG;
- GPU deployment.
