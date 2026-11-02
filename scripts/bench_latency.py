"""Latency / memory benchmark: network forward only, batch B (default 1), FP16, PyTorch eager, RTX 5070.

Usage: python scripts/bench_latency.py --family yolo --size s [--imgsz 640] [--h H --w W]
Prints one JSON line and appends it to results/latency.jsonl.
"""
import argparse, json, os, time
import numpy as np
import torch
from common import ROOT, load_yolo, load_rfdetr, gpu_info


def time_fn(fn, x, warmup=50, iters=300):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)), std_ms=float(ts.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--h", type=int); ap.add_argument("--w", type=int)
    ap.add_argument("--res", type=int, help="rfdetr square resolution override")
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--head", default="e2e", choices=["e2e", "nms"], help="YOLO26 head (see predict.py)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    if a.family == "yolo":
        y = load_yolo(a.size)
        y.model.model[-1].end2end = a.head == "e2e"  # e2e: fuse() then drops the one-to-many branch
        net = y.model.fuse().to(dev).half().eval()
        if a.head == "nms":
            from ultralytics.utils.nms import non_max_suppression
            fwd = net
            net_fn = lambda x: non_max_suppression(fwd(x)[0], 0.001, 0.7, max_det=300)
        else:
            net_fn = lambda x: net(x)
        h, w = a.h or 640, a.w or 640
        params = sum(p.numel() for p in net.parameters())
        fn = net_fn
        name = f"YOLO26-{a.size}" + ("-nms" if a.head == "nms" else "")
    else:
        r = load_rfdetr(a.size, a.res)
        r.inference(compile=False, dtype=torch.float16)
        net = r.model.inference_model
        res = r.model.resolution
        h, w = a.h or res, a.w or res
        params = sum(p.numel() for p in net.parameters())
        fn = lambda x: net(x)
        name = f"RF-DETR-{a.size}"
    x = torch.randn(a.batch, 3, h, w, device=dev, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t = time_fn(fn, x)
    out = dict(model=name, family=a.family, size=a.size, input_hw=[h, w], params_M=params / 1e6,
               peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20,
               activation_mem_MB=(torch.cuda.max_memory_allocated() - base_mem) / 2**20,
               dtype="fp16", batch=a.batch, tag=a.tag, per_image_ms=t["median_ms"] / a.batch, **t, **gpu_info())
    print(json.dumps(out))
    with open(os.path.join(ROOT, "results", "latency.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
