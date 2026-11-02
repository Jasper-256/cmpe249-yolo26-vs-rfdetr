#!/usr/bin/env bash
# Network-only latency/throughput/memory benchmark (FP16, PyTorch eager, synthetic input).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
rm -f results/latency.jsonl
for b in 1 16; do
  for s in n s m l x; do
    python scripts/bench_latency.py --family yolo --size $s --batch $b --head e2e
    python scripts/bench_latency.py --family yolo --size $s --batch $b --head nms
  done
  for s in nano small medium large; do python scripts/bench_latency.py --family rfdetr --size $s --batch $b; done
done
# KITTI-shaped inputs (what each pipeline actually feeds the network on a 1242x375 frame)
python scripts/bench_latency.py --family yolo --size m --h 224 --w 640  --tag kitti
python scripts/bench_latency.py --family yolo --size m --h 416 --w 1280 --tag kitti
python scripts/bench_latency.py --family rfdetr --size medium --h 320 --w 1056 --tag kitti
python scripts/bench_latency.py --family rfdetr --size medium --h 384 --w 1248 --tag kitti
