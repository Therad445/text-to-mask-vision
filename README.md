
# Text-to-Mask Vision

Open-vocabulary object detection and segmentation demo using Grounding DINO and SAM.

## Project idea

The system takes an image and a text prompt, detects the corresponding object using Grounding DINO, and refines the detected region into a segmentation mask with SAM.

Example:

```text
image + "dog" -> bounding box -> segmentation mask
```

## Main features

* text-guided object detection
* segmentation mask generation
* visualization of boxes and masks
* Streamlit demo
* latency measurement
* qualitative error analysis

## Project structure

```text
text-to-mask-vision/
├── app.py
├── src/
│   ├── grounding_dino_detector.py
│   ├── sam_segmenter.py
│   ├── pipeline.py
│   ├── visualization.py
│   └── utils.py
├── configs/
├── data/demo_images/
├── outputs/
├── notebooks/
├── report/
└── presentation/
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Grounding DINO installation may require additional setup depending on the environment. If local installation fails, run the first sanity-check in Google Colab.

## Run demo

```bash
streamlit run app.py
```

## Planned prompts

```text
person
dog
cat
car
bottle
chair
bicycle
bus
```

## Evaluation plan

We evaluate the system by:

* visual quality of detected boxes and masks
* inference latency
* sensitivity to prompts
* typical failure cases

## Limitations

The project uses pretrained models and does not fine-tune them. The main goal is to build a working end-to-end prototype and analyze its behavior.

## Future work

* compare Grounding DINO with OWL-ViT
* add threshold sweep experiments
* add more images and prompts
* add video support
* improve deployment packaging
  EOF

cat > requirements.txt <<'EOF'
torch
torchvision
numpy
opencv-python
matplotlib
Pillow
streamlit
segment-anything
supervision
transformers
timm
pycocotools
scipy
tqdm
pandas
