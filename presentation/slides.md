# Text-to-Mask Vision

Open-vocabulary object detection and segmentation with Grounding DINO + SAM

Radmir Islamov  
Computer Vision Final Project  
github.com/Therad445/text-to-mask-vision

---

# 1. Why this project matters

Classical detectors are usually fixed-class systems.

Text-to-mask pipeline:

```text
image + text prompt -> bounding boxes -> segmentation masks
```

Use cases:

- interactive annotation;
- dataset labeling;
- visual search;
- rapid prototyping.

---

# 2. End-to-end pipeline

```text
Input image + text prompt
        -> prompt normalization
        -> Grounding DINO detection
        -> NMS post-processing
        -> SAM box-guided segmentation
        -> visualization + detections table
```

Key idea: Grounding DINO localizes; SAM segments selected boxes.

---

# 3. Models used

**Grounding DINO**

- open-vocabulary object detection;
- input: image + text prompt;
- output: boxes, scores, labels.

**SAM**

- box-guided segmentation;
- input: image + selected boxes;
- output: masks and mask scores.

No training from scratch is performed.

---

# 4. Implementation and repository

Main files:

- `src/text_to_mask_pipeline.py`
- `app.py`
- `notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb`
- `notebooks/03_colab_batch_gallery.ipynb`
- `report/final_report.pdf`
- `scripts/check_project.sh`

Large checkpoints are not stored in Git.

---

# 5. Interactive demo

Streamlit demo supports:

- image upload;
- text prompt input;
- box/text threshold controls;
- NMS IoU threshold control;
- maximum detections control;
- result visualization;
- detections table.

Run:

```bash
python -m streamlit run app.py
```

---

# 6. Experiment design

Three experiment blocks:

1. sanity check;
2. local Streamlit demo;
3. batch gallery.

Batch gallery cases:

- crowded bears;
- bear cub prompt;
- bus/person multi-object prompt;
- clear person case;
- weak prompt case.

---

# 7. Success case: multi-object prompt

Prompt:

```text
bus . person .
```

Observations:

- bus detected with high confidence;
- several person instances detected;
- SAM produces masks from selected boxes;
- good example of open-vocabulary control.

---

# 8. Prompt sensitivity

Controlled comparison:

- same image;
- different prompts.

Correct prompt:

```text
bus . person .
```

Weak prompt:

```text
small animal .
```

The weak prompt produces lower-confidence and semantically wrong detections.

---

# 9. Crowded bears

Final tuned setting:

```text
prompt = bear .
box threshold = 0.25
text threshold = 0.30
NMS IoU = 0.35
max detections = 8
final detections = 5
```

Metrics:

```text
mean box score = 0.399
mean mask score = 0.971
```

Interpretation: the main bottleneck is detection/localization in crowded scenes.

---

# 10. Proxy metrics summary

Evaluation uses qualitative inspection and proxy metrics:

- raw detections;
- detections after NMS;
- mean box score;
- mean mask score;
- visual inspection.

A labeled benchmark with IoU/mAP is future work.

---

# 11. Individual contribution

Implemented and prepared:

- reusable text-to-mask pipeline;
- Streamlit demo;
- NMS post-processing;
- Colab sanity notebook;
- batch gallery notebook;
- experiment logs and visual results;
- final report and presentation.

---

# 12. Conclusion

The project implements a working text-to-mask CV system:

```text
image + text prompt -> open-vocabulary boxes -> masks -> interactive demo
```

The system is useful for interactive annotation, visual search and rapid prototyping.

Future work:

- labeled benchmark;
- OWL-ViT comparison;
- latency measurement;
- mask export;
- GPU deployment.
