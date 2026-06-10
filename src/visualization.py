from **future** import annotations

from PIL import Image, ImageDraw

def make_placeholder_visualization(image: Image.Image, text_prompt: str) -> Image.Image:
"""
Temporary visualization used before the real model pipeline is connected.
"""
result = image.copy()
draw = ImageDraw.Draw(result)
draw.rectangle((10, 10, result.width - 10, 50), outline="red", width=2)
draw.text((20, 22), f"Prompt: {text_prompt}", fill="red")
return result

def draw_boxes_and_masks(image, boxes, masks, phrases=None, scores=None):
"""
TODO: implement final visualization:
- draw bounding boxes
- overlay masks
- draw labels and confidence scores
"""
raise NotImplementedError("Final visualization is not implemented yet.")
