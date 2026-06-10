#!/usr/bin/env bash
set -e

mkdir -p weights

if [ -f "weights/sam_vit_b_01ec64.pth" ]; then
    echo "SAM checkpoint already exists: weights/sam_vit_b_01ec64.pth"
else
    echo "Downloading SAM ViT-B checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -O weights/sam_vit_b_01ec64.pth
fi

ls -lh weights
