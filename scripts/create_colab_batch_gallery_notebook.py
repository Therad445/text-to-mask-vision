import json
import textwrap
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


notebook = {
    "cells": [
        md(
            """
            # Experiment 03 — Batch Gallery for Text-to-Mask Pipeline

            This notebook runs a small qualitative gallery experiment for the project:

            ```text
            image + text prompt -> Grounding DINO boxes -> NMS -> SAM masks -> visualization
            ```

            The goal is to collect success cases, crowded/failure cases, and prompt sensitivity examples.
            """
        ),
        md("## 1. Check GPU"),
        code(
            """
            import torch

            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            DEVICE
            """
        ),
        md("## 2. Install dependencies"),
        code(
            """
            !pip install -q --upgrade transformers accelerate safetensors
            !pip install -q opencv-python matplotlib pillow numpy pandas torchvision
            !pip install -q git+https://github.com/facebookresearch/segment-anything.git
            """
        ),
        md("## 3. Download SAM checkpoint"),
        code(
            """
            !mkdir -p weights

            !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \\
                -O weights/sam_vit_b_01ec64.pth

            !ls -lh weights
            """
        ),
        md("## 4. Imports and model loading"),
        code(
            """
            from pathlib import Path
            import urllib.request
            import json

            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            import pandas as pd
            import torch
            from PIL import Image
            from torchvision.ops import nms
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from segment_anything import sam_model_registry, SamPredictor


            MODEL_ID = "IDEA-Research/grounding-dino-base"
            SAM_CHECKPOINT = "weights/sam_vit_b_01ec64.pth"
            SAM_MODEL_TYPE = "vit_b"

            DATA_DIR = Path("data/gallery_images")
            FIGURE_DIR = Path("report/figures/gallery")
            EXPERIMENT_DIR = Path("report/experiments")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

            processor = AutoProcessor.from_pretrained(MODEL_ID)
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
            grounding_model.eval()

            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            sam.to(device=DEVICE)

            sam_predictor = SamPredictor(sam)

            print("Models loaded")
            """
        ),
        md("## 5. Define test cases"),
        code(
            """
            CASES = [
                {
                    "case_id": "01_crowded_bears",
                    "url": "https://raw.githubusercontent.com/IDEA-Research/Grounded-Segment-Anything/main/assets/demo1.jpg",
                    "filename": "crowded_bears.jpg",
                    "prompt": "bear .",
                    "box_threshold": 0.25,
                    "text_threshold": 0.30,
                    "nms_iou_threshold": 0.35,
                    "max_detections": 8,
                    "note": "Crowded scene with visually overlapping bear instances.",
                },
                {
                    "case_id": "02_bear_cub_prompt",
                    "url": "https://raw.githubusercontent.com/IDEA-Research/Grounded-Segment-Anything/main/assets/demo1.jpg",
                    "filename": "crowded_bears.jpg",
                    "prompt": "bear cub .",
                    "box_threshold": 0.20,
                    "text_threshold": 0.20,
                    "nms_iou_threshold": 0.50,
                    "max_detections": 6,
                    "note": "Prompt sensitivity case on the same image.",
                },
                {
                    "case_id": "03_bus_person_multi_object",
                    "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
                    "filename": "bus.jpg",
                    "prompt": "bus . person .",
                    "box_threshold": 0.25,
                    "text_threshold": 0.25,
                    "nms_iou_threshold": 0.45,
                    "max_detections": 8,
                    "note": "Multi-object prompt with a bus and people.",
                },
                {
                    "case_id": "04_person_clear_case",
                    "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
                    "filename": "zidane.jpg",
                    "prompt": "person .",
                    "box_threshold": 0.25,
                    "text_threshold": 0.25,
                    "nms_iou_threshold": 0.45,
                    "max_detections": 4,
                    "note": "Clear person detection case.",
                },
                {
                    "case_id": "05_weak_prompt_case",
                    "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
                    "filename": "bus.jpg",
                    "prompt": "small animal .",
                    "box_threshold": 0.20,
                    "text_threshold": 0.20,
                    "nms_iou_threshold": 0.45,
                    "max_detections": 5,
                    "note": "Weak or mismatched prompt case.",
                },
            ]

            CASES
            """
        ),
        md("## 6. Helper functions"),
        code(
            """
            def download_file(url: str, output_path: Path) -> None:
                if output_path.exists() and output_path.stat().st_size > 0:
                    return

                print(f"Downloading {url} -> {output_path}")
                urllib.request.urlretrieve(url, output_path)


            def normalize_prompt(prompt: str) -> str:
                prompt = prompt.strip().lower()
                if not prompt.endswith("."):
                    prompt += " ."
                return prompt


            def show_mask(mask, ax, alpha: float = 0.45):
                mask = mask.detach().cpu().numpy()

                if mask.ndim == 3:
                    mask = mask[0]

                color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask_image)


            def show_box(box, ax, label: str | None = None):
                x0, y0, x1, y1 = box.detach().cpu().numpy()

                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=2,
                )
                ax.add_patch(rect)

                if label is not None:
                    ax.text(
                        x0,
                        max(y0 - 5, 0),
                        label,
                        color="white",
                        fontsize=10,
                        bbox=dict(facecolor="green", alpha=0.7),
                    )


            def run_text_to_mask_case(case: dict) -> dict:
                image_path = DATA_DIR / case["filename"]
                download_file(case["url"], image_path)

                image_pil = Image.open(image_path).convert("RGB")
                image_np = np.array(image_pil)
                prompt = normalize_prompt(case["prompt"])

                inputs = processor(
                    images=image_pil,
                    text=prompt,
                    return_tensors="pt",
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=case["box_threshold"],
                    text_threshold=case["text_threshold"],
                    target_sizes=[image_pil.size[::-1]],
                )

                result = results[0]

                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                labels = result.get("labels", result.get("text_labels", []))

                raw_detections = len(boxes)

                if len(boxes) > 0:
                    keep = nms(
                        boxes,
                        scores,
                        iou_threshold=case["nms_iou_threshold"],
                    )
                    keep = keep[: case["max_detections"]]

                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = [labels[i] for i in keep.tolist()]

                sam_predictor.set_image(image_np)

                if len(boxes) > 0:
                    boxes_torch = boxes.to(DEVICE)
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                        boxes_torch,
                        image_np.shape[:2],
                    ).to(DEVICE)

                    with torch.no_grad():
                        masks, mask_scores, _ = sam_predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,
                        )

                    masks = masks.detach().cpu()
                    mask_scores = mask_scores.detach().cpu()
                else:
                    masks = torch.empty((0, 1, image_np.shape[0], image_np.shape[1]))
                    mask_scores = torch.empty((0, 1))

                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(image_pil)

                for i, box in enumerate(boxes):
                    label = labels[i] if i < len(labels) else "object"
                    score = float(scores[i]) if i < len(scores) else 0.0
                    show_box(box, ax, label=f"{label}: {score:.2f}")

                    if i < len(masks):
                        show_mask(masks[i], ax)

                ax.axis("off")
                ax.set_title(f"{case['case_id']} | prompt: {prompt}")

                output_path = FIGURE_DIR / f"{case['case_id']}.png"
                fig.savefig(output_path, bbox_inches="tight", dpi=150)
                plt.show()
                plt.close(fig)

                return {
                    "case_id": case["case_id"],
                    "image": case["filename"],
                    "prompt": prompt,
                    "box_threshold": case["box_threshold"],
                    "text_threshold": case["text_threshold"],
                    "nms_iou_threshold": case["nms_iou_threshold"],
                    "max_detections": case["max_detections"],
                    "raw_detections": raw_detections,
                    "detections_after_nms": len(boxes),
                    "box_scores": ", ".join(f"{float(x):.4f}" for x in scores),
                    "mask_scores": ", ".join(f"{float(x):.4f}" for x in mask_scores.flatten()),
                    "labels": ", ".join(labels),
                    "output_path": str(output_path),
                    "note": case["note"],
                }
            """
        ),
        md("## 7. Run batch gallery"),
        code(
            """
            rows = []

            for case in CASES:
                print("=" * 80)
                print("Running:", case["case_id"], "| prompt:", case["prompt"])
                row = run_text_to_mask_case(case)
                rows.append(row)

            results_df = pd.DataFrame(rows)
            results_path = EXPERIMENT_DIR / "03_gallery_results.csv"
            results_df.to_csv(results_path, index=False)

            print("Saved:", results_path)
            results_df
            """
        ),
        md("## 8. Save a markdown summary"),
        code(
            """
            summary_path = EXPERIMENT_DIR / "03_gallery_summary.md"

            lines = [
                "# Experiment 03 — Result Gallery and Prompt Sensitivity",
                "",
                "## Goal",
                "",
                "This experiment evaluates the text-to-mask pipeline on several qualitative cases:",
                "",
                "- crowded scenes;",
                "- multi-object prompts;",
                "- clear object cases;",
                "- prompt sensitivity;",
                "- weak or mismatched prompts.",
                "",
                "## Results",
                "",
                "| Case | Prompt | Raw detections | After NMS | Notes |",
                "|---|---|---:|---:|---|",
            ]

            for row in rows:
                lines.append(
                    f"| {row['case_id']} | `{row['prompt']}` | "
                    f"{row['raw_detections']} | {row['detections_after_nms']} | "
                    f"{row['note']} |"
                )

            lines.extend(
                [
                    "",
                    "## Observation",
                    "",
                    "The pipeline works well on clear objects and multi-object prompts. "
                    "Crowded scenes remain challenging because Grounding DINO can produce duplicate or overlapping boxes, "
                    "and NMS can suppress nearby valid detections.",
                    "",
                    "The weak prompt case illustrates that open-vocabulary detection is sensitive to prompt wording.",
                ]
            )

            summary_path.write_text("\\n".join(lines), encoding="utf-8")
            print(summary_path.read_text(encoding="utf-8"))
            """
        ),
        md("## 9. Package artifacts for download"),
        code(
            """
            !zip -r gallery_artifacts.zip report/figures/gallery report/experiments/03_gallery_results.csv report/experiments/03_gallery_summary.md

            print("Created gallery_artifacts.zip")
            print("Download it from the Colab file browser or use the next cell.")
            """
        ),
        md("## 10. Optional: download artifacts"),
        code(
            """
            from google.colab import files
            files.download("gallery_artifacts.zip")
            """
        ),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks/03_colab_batch_gallery.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Created {output_path}")
