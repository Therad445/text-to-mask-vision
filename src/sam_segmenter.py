from **future** import annotations

class SAMSegmenter:
"""
Wrapper for Segment Anything Model.

```
Expected responsibility:
- load SAM checkpoint
- receive image and boxes
- return segmentation masks
"""

def __init__(self, checkpoint_path: str | None = None, device: str = "cuda") -> None:
    self.checkpoint_path = checkpoint_path
    self.device = device

def predict_masks(self, image, boxes):
    raise NotImplementedError("SAM wrapper is not implemented yet.")
```

