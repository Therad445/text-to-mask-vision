# Text-to-Mask Vision

Open-vocabulary object detection and segmentation pipeline based on **Grounding DINO** and **Segment Anything Model (SAM)**.

The project implements a practical text-to-mask computer vision system:

```text
image + text prompt -> bounding boxes -> segmentation masks
```

The user uploads an image and provides a text prompt such as `bear .`, `person .`, or `bus . person .`. Grounding DINO detects objects matching the prompt, non-maximum suppression removes duplicate boxes, and SAM converts the selected boxes into segmentation masks.

## Project Format

This is a practical Computer Vision final project.

The project includes:

* reusable inference pipeline;
* Streamlit web demo;
* Google Colab experiments;
* qualitative gallery of success and failure cases;
* CSV experiment logs;
* final PDF report;
* presentation draft;
* error analysis and future work.

## Project Goal

The goal is to build a working prototype of an interactive text-to-mask system that can be used for:

* interactive image annotation;
* dataset labeling;
* visual search;
* rapid prototyping of computer vision pipelines;
* prompt-based segmentation without training a new model.

## Method

The pipeline consists of four main stages:

1. Load an input image and a text prompt.
2. Use Grounding DINO for open-vocabulary object detection.
3. Apply non-maximum suppression to reduce duplicate boxes.
4. Use SAM to generate segmentation masks from the selected boxes.

```text
Input image + text prompt
        |
        v
Grounding DINO
        |
        v
Bounding boxes + scores
        |
        v
NMS post-processing
        |
        v
SAM box-guided segmentation
        |
        v
Masks + visualization + detections table
```

## Models

* Detection model: `IDEA-Research/grounding-dino-base`
* Detection interface: Hugging Face Transformers
* Segmentation model: Segment Anything Model, ViT-B
* SAM checkpoint: `sam_vit_b_01ec64.pth`

Large model checkpoints are not stored in Git. The `weights/` directory is ignored.

## Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── configs/
│   └── prompts.yaml
├── src/
│   ├── text_to_mask_pipeline.py
│   ├── benchmark.py
│   └── utils.py
├── notebooks/
│   ├── 02_colab_hf_grounding_dino_sam_sanity_check.ipynb
│   └── 03_colab_batch_gallery.ipynb
├── report/
│   ├── final_report.md
│   ├── final_report.pdf
│   ├── experiments/
│   │   ├── 01_sanity_check.md
│   │   ├── 02_streamlit_local_demo.md
│   │   ├── 03_gallery_plan.md
│   │   ├── 03_gallery_results.csv
│   │   ├── 03_gallery_summary.md
│   │   └── 04_evaluation_summary.md
│   └── figures/
│       ├── sanity_text_to_mask_hf.png
│       ├── streamlit_demo_result.png
│       ├── streamlit_detections_table.png
│       └── gallery/
├── presentation/
│   └── slides.md
└── scripts/
    ├── check_project.sh
    ├── create_colab_batch_gallery_notebook.py
    ├── download_sam_weights.sh
    └── run_demo.sh
```

## Demo

The project includes a Streamlit application:

```bash
python -m streamlit run app.py
```

Before running the app locally, download the SAM checkpoint:

```bash
./scripts/download_sam_weights.sh
```

For CPU-only machines, local inference can be slow. GPU inference via Google Colab is recommended for full experiments.

## Experiments

The project contains three main experiment blocks.

### Experiment 01: Sanity Check

Notebook:

```text
notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb
```

This experiment verifies the full pipeline on a bear image:

```text
image -> Grounding DINO -> NMS -> SAM -> visualization
```

### Experiment 02: Local Streamlit Demo

The local demo verifies that the project works as an interactive web application.

The demo supports:

* image upload;
* text prompt input;
* box threshold;
* text threshold;
* NMS IoU threshold;
* maximum detections;
* output visualization;
* detections table.

### Experiment 03: Batch Gallery

Notebook:

```text
notebooks/03_colab_batch_gallery.ipynb
```

The batch gallery tests several qualitative cases:

| Case              | Prompt           | Purpose                          |
| ----------------- | ---------------- | -------------------------------- |
| Crowded bears     | `bear .`         | crowded scene / failure analysis |
| Bear cub prompt   | `bear cub .`     | prompt sensitivity               |
| Bus and people    | `bus . person .` | multi-object prompt              |
| Clear person case | `person .`       | simple success case              |
| Weak prompt case  | `small animal .` | mismatched prompt / limitation   |

The experiment produces:

* gallery images;
* CSV results;
* qualitative summary;
* proxy evaluation summary.

## Evaluation Summary

The project does not use a manually labeled segmentation benchmark. Therefore, evaluation is based on qualitative inspection and proxy metrics from the detection and segmentation pipeline.

| Case                         | Raw detections | After NMS | Mean box score | Mean mask score | Interpretation                                               |
| ---------------------------- | -------------: | --------: | -------------: | --------------: | ------------------------------------------------------------ |
| `01_crowded_bears`           |              8 |         4 |          0.438 |           0.951 | Crowded scene; NMS reduces duplicate/overlapping detections. |
| `02_bear_cub_prompt`         |              5 |         4 |          0.425 |           0.958 | More specific prompt; useful for prompt sensitivity.         |
| `03_bus_person_multi_object` |              5 |         5 |          0.709 |           0.962 | Successful multi-object prompt.                              |
| `04_person_clear_case`       |              2 |         2 |          0.654 |           0.987 | Clear simple case.                                           |
| `05_weak_prompt_case`        |              5 |         5 |          0.286 |           0.961 | Mismatched prompt; lower confidence.                         |

Key observations:

* the `bus . person .` prompt has the strongest mean box confidence;
* the `small animal .` prompt has much lower box confidence;
* SAM mask scores can remain high even for semantically weak detections, because SAM segments boxes provided by the detector;
* detection quality is the main bottleneck;
* crowded scenes remain difficult because nearby objects overlap and NMS may suppress valid instances.

## Results

The implemented system successfully demonstrates an end-to-end text-to-mask pipeline.

The strongest results are observed when:

* the target object is clearly visible;
* the prompt is simple and specific;
* objects do not heavily overlap;
* the image contains common visual categories.

The system also supports multi-object prompts such as:

```text
bus . person .
```

## Limitations

Observed limitations:

* CPU inference is slow;
* crowded scenes are difficult;
* close objects may be merged or suppressed by NMS;
* segmentation quality depends on detection box quality;
* prompt wording strongly affects results;
* weak or mismatched prompts can produce semantically incorrect detections.

The crowded bears example is intentionally included as an error-analysis case. Not all bear instances are perfectly separated, because small nearby animals overlap strongly and can be suppressed during post-processing.

The bus/person and weak prompt cases intentionally use the same image to demonstrate prompt sensitivity under controlled visual input.

## Final Artifacts

Important files:

* final report: `report/final_report.pdf`
* report source: `report/final_report.md`
* presentation: `presentation/slides.md`
* Streamlit app: `app.py`
* main pipeline: `src/text_to_mask_pipeline.py`
* sanity notebook: `notebooks/02_colab_hf_grounding_dino_sam_sanity_check.ipynb`
* batch gallery notebook: `notebooks/03_colab_batch_gallery.ipynb`
* gallery results: `report/experiments/03_gallery_results.csv`
* evaluation summary: `report/experiments/04_evaluation_summary.md`

## Project Check

Run:

```bash
bash scripts/check_project.sh
```

Expected output:

```text
Checking project structure...
Checking that large/local files are not tracked...
Checking for unfinished placeholder code...
Project check passed.
```

## Future Work

Possible improvements:

* compare Grounding DINO with OWL-ViT;
* add a labeled benchmark and compute IoU / precision / recall;
* measure CPU and GPU latency;
* add batch inference to the Streamlit app;
* export masks as PNG or COCO annotations;
* deploy the demo on a GPU-enabled server;
* add mask editing tools.

