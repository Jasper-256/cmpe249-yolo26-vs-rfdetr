#!/usr/bin/env bash
# Inference on COCO val2017 (5000 images) for every YOLO26 size (both heads) and every RF-DETR size.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
for s in n s m l x; do python scripts/predict.py --family yolo --size $s --dataset coco --head e2e; done
for s in n s m l x; do python scripts/predict.py --family yolo --size $s --dataset coco --head nms; done
for s in nano small medium large; do python scripts/predict.py --family rfdetr --size $s --dataset coco; done
python scripts/evaluate.py coco results/dets/coco_*.json | tee results/coco_summary.txt
