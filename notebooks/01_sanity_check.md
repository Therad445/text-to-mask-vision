
# Sanity Check Notebook Plan

Copy these cells into Colab or Jupyter.

## Cell 1. Check GPU

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 2. Install dependencies

```python
!pip install -q torch torchvision opencv-python matplotlib pillow numpy
!pip install -q segment-anything
!pip install -q groundingdino-py
```

## Cell 3. Load image

```python
from PIL import Image
import matplotlib.pyplot as plt

image_path = "demo.jpg"
image = Image.open(image_path).convert("RGB")

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
```

## Cell 4. Run Grounding DINO

Goal:

```text
image + text prompt -> boxes
```

## Cell 5. Run SAM

Goal:

```text
image + boxes -> masks
```

## Cell 6. Save final visualization

Goal:

```text
outputs/predictions/example_001.png
```

