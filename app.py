from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.pipeline import run_text_to_mask

st.set_page_config(
page_title="Text-to-Mask Vision",
layout="wide",
)

st.title("Text-to-Mask Vision")
st.write(
"Open-vocabulary object detection and segmentation demo using "
"Grounding DINO and SAM."
)

uploaded_file = st.file_uploader(
"Upload an image",
type=["jpg", "jpeg", "png"],
)

prompt = st.text_input(
"Text prompt",
value="person",
)

col1, col2 = st.columns(2)

with col1:
box_threshold = st.slider(
"Box threshold",
min_value=0.05,
max_value=0.90,
value=0.30,
step=0.05,
)

with col2:
text_threshold = st.slider(
"Text threshold",
min_value=0.05,
max_value=0.90,
value=0.25,
step=0.05,
)

run_button = st.button("Run text-to-mask pipeline")

if uploaded_file is not None:
image = Image.open(uploaded_file).convert("RGB")
st.subheader("Input image")
st.image(image, use_container_width=True)

if run_button:
if uploaded_file is None:
st.warning("Please upload an image first.")
elif not prompt.strip():
st.warning("Please enter a text prompt.")
else:
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
image.save(tmp.name)
image_path = tmp.name

```
    try:
        result = run_text_to_mask(
            image_path=image_path,
            text_prompt=prompt.strip(),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        st.subheader("Result")
        st.image(result["visualization"], use_container_width=True)

        st.write("Detected phrases:", result.get("phrases", []))
        st.write("Number of detections:", result.get("num_detections", 0))
        st.write("Total latency, sec:", round(result.get("latency_total", 0.0), 3))

    except NotImplementedError as exc:
        st.error(str(exc))
        st.info(
            "The project skeleton is ready. Next step: implement "
            "Grounding DINO and SAM wrappers in src/."
        )
    except Exception as exc:
        st.exception(exc)
```

