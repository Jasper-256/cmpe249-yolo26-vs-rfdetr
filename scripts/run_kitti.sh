#!/usr/bin/env bash
# Zero-shot inference on KITTI 2D object training split (7481 images, no retraining) for every size,
# plus the input-resolution / aspect-ratio ablation on the medium pair.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/make_kitti_gt.py
for s in n s m l x; do python scripts/predict.py --family yolo --size $s --dataset kitti --head e2e; done
for s in n s m l x; do python scripts/predict.py --family yolo --size $s --dataset kitti --head nms; done
for s in nano small medium large; do python scripts/predict.py --family rfdetr --size $s --dataset kitti; done
# ablation: input resolution / aspect ratio (medium pair)
python scripts/predict.py --family yolo   --size m      --dataset kitti --imgsz 1280
python scripts/predict.py --family rfdetr --size medium --dataset kitti --autocast           # control: square 576, same autocast code path
python scripts/predict.py --family rfdetr --size medium --dataset kitti --shape 320 1056   # ~same pixels as 576^2, aspect kept
python scripts/predict.py --family rfdetr --size medium --dataset kitti --shape 384 1248   # ~native KITTI resolution
python scripts/evaluate.py kitti results/dets/kitti_*.json | tee results/kitti_summary.txt
