from **future** import annotations

import time
from typing import Any

from PIL import Image

from src.visualization import make_placeholder_visualization

def run_text_to_mask(
image_path: str,
text_prompt: str,
box_threshold: float = 0.30,
text_threshold: float = 0.25,
) -> dict[str, Any]:
"""
Main project pipeline.

```
Planned flow:
1. Load image.
2. Run Grounding DINO with text prompt.
3. Get bounding boxes and confidence scores.
4. Run SAM using boxes as prompts.
5. Draw boxes and masks.
6. Return result and latency.

Current state:
placeholder implementation.
"""

start_time = time.perf_counter()
image = Image.open(image_path).convert("RGB")

visualization = make_placeholder_visualization(
    image=image,
    text_prompt=text_prompt,
)

latency_total = time.perf_counter() - start_time

raise NotImplementedError(
    "Grounding DINO + SAM is not implemented yet. "
    "Fill src/grounding_dino_detector.py, src/sam_segmenter.py, "
    "and then update src/pipeline.py."
)

return {
    "visualization": visualization,
    "boxes": [],
    "scores": [],
    "phrases": [],
    "masks": [],
    "num_detections": 0,
    "latency_total": latency_total,
}
```

