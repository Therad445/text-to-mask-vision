from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.patches as patches

from src.text_to_mask_pipeline import TextToMaskPipeline


SAM_CHECKPOINT = "weights/sam_vit_b_01ec64.pth"


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


def visualize_result(image: Image.Image, result, title: str):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    for i, box in enumerate(result.boxes):
        label = result.labels[i] if i < len(result.labels) else "object"
        score = float(result.scores[i]) if i < len(result.scores) else 0.0

        show_box(box, ax, label=f"{label}: {score:.2f}")

        if i < len(result.masks):
            show_mask(result.masks[i], ax)

    ax.axis("off")
    ax.set_title(title)

    return fig


@st.cache_resource
def load_pipeline():
    return TextToMaskPipeline(
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam_checkpoint=SAM_CHECKPOINT,
        sam_model_type="vit_b",
    )


st.set_page_config(
    page_title="Text-to-Mask Vision Demo",
    page_icon="🧩",
    layout="wide",
)

st.title("Text-to-Mask Vision Demo")
st.write(
    "Open-vocabulary object detection with Grounding DINO "
    "followed by SAM-based segmentation."
)

checkpoint_path = Path(SAM_CHECKPOINT)

if not checkpoint_path.exists():
    st.error(
        "SAM checkpoint not found. Run this command first:\n\n"
        "`./scripts/download_sam_weights.sh`"
    )
    st.stop()

with st.sidebar:
    st.header("Settings")

    text_prompt = st.text_input(
        "Text prompt",
        value="bear .",
        help="Use lowercase dot-separated prompts, for example: `bear .`, `dog .`, `person . car .`",
    )

    box_threshold = st.slider(
        "Box threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
    )

    text_threshold = st.slider(
        "Text threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.30,
        step=0.05,
    )

    nms_iou_threshold = st.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    max_detections = st.slider(
        "Max detections",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to run the text-to-mask pipeline.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

run_button = st.button("Run text-to-mask pipeline", type="primary")

if run_button:
    with st.spinner("Loading models and running inference..."):
        pipeline = load_pipeline()

        result = pipeline.run(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

    with right_col:
        st.subheader("Result")

        if len(result.boxes) == 0:
            st.warning("No objects detected. Try lowering thresholds or changing the prompt.")
        else:
            fig = visualize_result(
                image=image,
                result=result,
                title=f"Text-to-mask result: {text_prompt}",
            )
            st.pyplot(fig)

    st.subheader("Detections")

    if len(result.boxes) > 0:
        rows = []

        for i, box in enumerate(result.boxes):
            mask_score = (
                float(result.mask_scores[i])
                if i < len(result.mask_scores)
                else None
            )

            rows.append(
                {
                    "label": result.labels[i] if i < len(result.labels) else "object",
                    "box_score": float(result.scores[i]),
                    "mask_score": mask_score,
                    "x0": float(box[0]),
                    "y0": float(box[1]),
                    "x1": float(box[2]),
                    "y1": float(box[3]),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.caption(
            "Note: Grounding DINO may return overlapping boxes in crowded scenes. "
            "NMS is used to reduce duplicates before passing boxes to SAM."
        )
