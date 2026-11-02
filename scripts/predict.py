"""Run a detector over a dataset at batch 1 (FP16) and dump COCO-format detections + per-image timing.

Usage:
  python scripts/predict.py --family yolo   --size m      --dataset coco
  python scripts/predict.py --family rfdetr --size medium --dataset kitti [--res 640] [--imgsz 1280]

Detections use COCO-91 category ids for both families. Confidence threshold 0.001 for AP computation,
at most 300 detections/image. By default no NMS is applied by either pipeline: YOLO26 uses its end-to-end
(one-to-one) head (--head e2e), RF-DETR its set-prediction top-K postprocess. --head nms selects YOLO26's
one-to-many head + NMS (IoU 0.7) for comparison.
"""
import argparse, glob, json, os, time
import cv2
import numpy as np
import torch
from common import ROOT, load_yolo, load_rfdetr, gpu_info

CONF = 0.001
MAX_DET = 300


def list_images(dataset):
    if dataset == "coco":
        ann = json.load(open(os.path.join(ROOT, "data/coco/annotations/instances_val2017.json")))
        return [(im["id"], os.path.join(ROOT, "data/coco/val2017", im["file_name"])) for im in ann["images"]]
    files = sorted(glob.glob(os.path.join(ROOT, "data/kitti/training/image_2/*.png")))
    return [(int(os.path.basename(f)[:-4]), f) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--dataset", required=True, choices=["coco", "kitti"])
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO letterbox long side")
    ap.add_argument("--head", default="e2e", choices=["e2e", "nms"],
                    help="YOLO26 head: e2e = one-to-one NMS-free head (nms=False); "
                         "nms = one-to-many head + class-wise NMS @ IoU 0.7 (Ultralytics default)")
    ap.add_argument("--res", type=int, default=None, help="RF-DETR square input resolution override")
    ap.add_argument("--shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="RF-DETR non-square input shape (each divisible by 32), e.g. 384 1248 for KITTI")
    ap.add_argument("--autocast", action="store_true",
                    help="RF-DETR: run the un-optimised model under torch.autocast(fp16) instead of the "
                         "fp16 'inference()' export model (required for non-square --shape)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    # cudnn.benchmark would re-autotune for every new letterbox shape (YOLO rect inference) -> keep off
    torch.backends.cudnn.benchmark = False

    images = list_images(a.dataset)
    if a.limit:
        images = images[: a.limit]

    if a.family == "yolo":
        from ultralytics.data.converter import coco80_to_coco91_class
        to91 = coco80_to_coco91_class()
        model = load_yolo(a.size)
        name = f"yolo26{a.size}" + ("-nms" if a.head == "nms" else "") + (f"@{a.imgsz}" if a.imgsz != 640 else "")
        nms_arg = False if a.head == "e2e" else None

        def run(img_bgr):
            r = model.predict(img_bgr, imgsz=a.imgsz, conf=CONF, max_det=MAX_DET, quantize="fp16",
                              nms=nms_arg, device=0, verbose=False)[0]
            b = r.boxes
            return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), np.array([to91[int(c)] for c in b.cls.cpu().numpy()])
    else:
        model = load_rfdetr(a.size, a.res)
        use_autocast = a.autocast or a.shape is not None
        if not use_autocast:  # optimised export model is locked to a square resolution
            model.inference(compile=False, dtype=torch.float16)
        name = (f"rfdetr-{a.size}" + (f"@{a.res}" if a.res else "") + (f"@{a.shape[0]}x{a.shape[1]}" if a.shape else "")
                + ("-autocast" if a.autocast and not a.shape else ""))
        shape = tuple(a.shape) if a.shape else None

        def run(img_bgr):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
                d = model.predict(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), threshold=CONF, shape=shape,
                              include_source_image=False)
            return d.xyxy, d.confidence, d.class_id  # class_id is the sparse COCO-91 id for COCO checkpoints

    if a.tag:
        name += f"_{a.tag}"
    dets, times = [], []
    # warmup
    warm = cv2.imread(images[0][1])
    for _ in range(20):
        run(warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for i, (img_id, path) in enumerate(images):
        img = cv2.imread(path)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        xyxy, score, cls = run(img)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        order = np.argsort(-score)[:MAX_DET]
        for k in order:
            x1, y1, x2, y2 = [float(v) for v in xyxy[k]]
            dets.append(dict(image_id=img_id, category_id=int(cls[k]), bbox=[x1, y1, x2 - x1, y2 - y1],
                             score=float(score[k])))
        if i % 1000 == 0:
            print(f"{name} {a.dataset}: {i}/{len(images)}  median e2e {np.median(times):.2f} ms", flush=True)
    t = np.array(times)
    meta = dict(model=name, dataset=a.dataset, n_images=len(images), conf=CONF, max_det=MAX_DET,
                imgsz=a.imgsz if a.family == "yolo" else (a.shape or a.res or model.model.resolution),
                e2e_median_ms=float(np.median(t)), e2e_mean_ms=float(t.mean()), e2e_p90_ms=float(np.percentile(t, 90)),
                peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20, wall_s=time.time() - t_start, **gpu_info())
    os.makedirs(os.path.join(ROOT, "results/dets"), exist_ok=True)
    json.dump(dets, open(os.path.join(ROOT, f"results/dets/{a.dataset}_{name}.json"), "w"))
    json.dump(meta, open(os.path.join(ROOT, f"results/dets/{a.dataset}_{name}.meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
