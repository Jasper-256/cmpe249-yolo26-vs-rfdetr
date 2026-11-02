# CMPE 249 HW — 2D Object Detection, Track A: YOLO26 vs RF-DETR

Repository: https://github.com/Jasper-256/cmpe249-yolo26-vs-rfdetr

The report is **[`report/REPORT.md`](report/REPORT.md)**. The figures it uses are in `report/figs/`.

Layout:

| Path | Contents |
|---|---|
| `scripts/predict.py` | Runs one detector over COCO val2017 or KITTI and dumps COCO-format detections with timing metadata |
| `scripts/evaluate.py` | pycocotools scoring. For KITTI: taxonomy mapping, cyclist composite, KITTI-style AP_R40 |
| `scripts/make_kitti_gt.py` | Converts KITTI labels to COCO format (Van/Person_sitting/DontCare handled as ignore) |
| `scripts/bench_latency.py`, `run_bench.sh` | Network latency / throughput / memory benchmark |
| `scripts/run_coco.sh`, `run_kitti.sh` | Every accuracy experiment |
| `scripts/crowd_analysis.py`, `kitti_errors.py`, `aspect_diag.py`, `class_diag.py` | The analyses in report §6–7 |
| `scripts/inspect_shapes.py` | Forward-hook tensor shapes, used for the architecture diagrams |
| `scripts/analyze.py`, `visualize.py` | Tables (`results/tables.md`) and figures |
| `results/` | Metrics JSON per run (`metrics/`), summaries, `latency.jsonl`, `shapes.txt`. Raw detection dumps (`dets/`, 3.9 GB) are git-ignored |
| `logs/` | Raw run logs |
| `requirements-lock.txt` | Exact package versions |

Quick start (full commands are in report Appendix A):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-lock.txt
bash scripts/run_bench.sh && bash scripts/run_coco.sh && bash scripts/run_kitti.sh && python scripts/analyze.py
```

Build the PDF (`report/REPORT.pdf`). This uses the system Google Chrome through Playwright:

```bash
uv run --no-project --python 3.12 --with markdown-it-py --with mdit-py-plugins --with playwright python scripts/build_pdf.py
```
