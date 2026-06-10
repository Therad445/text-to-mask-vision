from **future** import annotations

class GroundingDINODetector:
"""
Wrapper for Grounding DINO.

```
Expected responsibility:
- load Grounding DINO model
- run text-guided detection
- return boxes, scores, phrases
"""

def __init__(self, device: str = "cuda") -> None:
    self.device = device

def predict(
    self,
    image,
    text_prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
):
    raise NotImplementedError("Grounding DINO wrapper is not implemented yet.")
```

