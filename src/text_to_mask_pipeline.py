from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from segment_anything import SamPredictor, sam_model_registry


@dataclass
class DetectionResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: List[str]
    masks: torch.Tensor
    mask_scores: torch.Tensor


class TextToMaskPipeline:
    def __init__(
        self,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth",
        sam_model_type: str = "vit_b",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_id
        ).to(self.device)
        self.grounding_model.eval()

        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if not text_prompt.strip().endswith("."):
            text_prompt = text_prompt.strip() + " ."

        text_prompt = text_prompt.lower()

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )

        result = results[0]

        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        labels = result["labels"]

        if len(boxes) == 0:
            return boxes, scores, labels

        keep = nms(boxes, scores, iou_threshold=nms_iou_threshold)
        keep = keep[:max_detections]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep.tolist()]

        return boxes, scores, labels

    def segment(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(boxes) == 0:
            empty_masks = torch.empty((0, 1, image.height, image.width))
            empty_scores = torch.empty((0, 1))
            return empty_masks, empty_scores

        image_np = np.array(image.convert("RGB"))
        self.sam_predictor.set_image(image_np)

        boxes_torch = boxes.to(self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_torch,
            image_np.shape[:2],
        ).to(self.device)

        with torch.no_grad():
            masks, mask_scores, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        return masks.detach().cpu(), mask_scores.detach().cpu()

    def run(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 4,
    ) -> DetectionResult:
        image = image.convert("RGB")

        boxes, scores, labels = self.detect(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

        masks, mask_scores = self.segment(image=image, boxes=boxes)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=masks,
            mask_scores=mask_scores,
        )
