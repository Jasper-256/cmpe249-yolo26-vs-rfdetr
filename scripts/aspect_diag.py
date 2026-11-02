"""Diagnose RF-DETR's behaviour on non-square (aspect-preserving) inputs.

1) KITTI: for Car detections (score >= 0.5) matched to a GT Car at IoU >= 0.5, report median predicted/GT
   width and height ratios for each input-shape variant (uses existing dumps in results/dets).
2) COCO: on 4:3 landscape val images (640x480), compare RF-DETR-M AP at the default square 576x576
   vs aspect-preserving 448x608 (both under fp16 autocast), to show the effect is not KITTI-specific.
Usage: python scripts/aspect_diag.py
"""
import contextlib, io, json, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, load_rfdetr
from kitti_errors import load_gt, iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def kitti_ratios(name, gt):
    dets = json.load(open(f"{ROOT}/results/dets/kitti_{name}.json"))
    by = {}
    for d in dets:
        if d["category_id"] == 3 and d["score"] >= 0.5:
            x, y, w, h = d["bbox"]; by.setdefault(d["image_id"], []).append([x, y, x + w, y + h])
    rw, rh, dy_top, dy_bot = [], [], [], []
    for iid, objs in gt.items():
        G = np.array([b for c, b in objs if c == "Car"]).reshape(-1, 4)
        for b in by.get(iid, []):
            b = np.array(b); ov = iou(b, G)
            if len(ov) and ov.max() >= 0.5:
                g = G[ov.argmax()]
                rw.append((b[2] - b[0]) / (g[2] - g[0])); rh.append((b[3] - b[1]) / (g[3] - g[1]))
                dy_top.append(b[1] - g[1]); dy_bot.append(b[3] - g[3])
    return dict(n=len(rw), w_ratio=float(np.median(rw)), h_ratio=float(np.median(rh)),
                top_shift_px=float(np.median(dy_top)), bottom_shift_px=float(np.median(dy_bot)))


def coco_aspect(shapes, n=1000):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(f"{ROOT}/data/coco/annotations/instances_val2017.json")
    ids = [i["id"] for i in gt.dataset["images"] if i["width"] == 640 and i["height"] == 480][:n]
    m = load_rfdetr("medium")
    res = {}
    for shape in shapes:
        dets = []
        for iid in ids:
            img = cv2.cvtColor(cv2.imread(f"{ROOT}/data/coco/val2017/{gt.imgs[iid]['file_name']}"), cv2.COLOR_BGR2RGB)
            with torch.autocast("cuda", dtype=torch.float16):
                d = m.predict(img, threshold=0.001, shape=shape, include_source_image=False)
            for b, s, c in zip(d.xyxy, d.confidence, d.class_id):
                dets.append(dict(image_id=iid, category_id=int(c), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], score=float(s)))
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, gt.loadRes(dets), "bbox"); E.params.imgIds = ids
            E.evaluate(); E.accumulate(); E.summarize()
        res[f"{shape[0]}x{shape[1]}"] = dict(AP=float(E.stats[0]), AP50=float(E.stats[1]), AP75=float(E.stats[2]))
        print("COCO 640x480 subset", shape, res[f"{shape[0]}x{shape[1]}"], flush=True)
    return dict(n_images=len(ids), results=res)


if __name__ == "__main__":
    gt = load_gt()
    out = {"kitti_car_box_ratios": {}}
    for name in ["rfdetr-medium", "rfdetr-medium@320x1056", "rfdetr-medium@384x1248", "yolo26m", "yolo26m@1280"]:
        out["kitti_car_box_ratios"][name] = kitti_ratios(name, gt)
        print(name, out["kitti_car_box_ratios"][name], flush=True)
    out["coco_4x3_subset"] = coco_aspect([(576, 576), (448, 608), (480, 640)])
    json.dump(out, open(f"{ROOT}/results/metrics/aspect_diag.json", "w"), indent=1)
