"""Two small diagnostics behind the cross-dataset per-class analysis.
 (1) Which COCO class do high-confidence (>=0.3) detections overlapping a KITTI Tram / Truck / Pedestrian / Car GT
     (IoU >= 0.5) carry?  -> taxonomy-mismatch evidence.
 (2) Median predicted/GT width and height ratio for matched Pedestrians (IoU >= 0.5, score >= 0.3)
     -> annotation-convention evidence (why Pedestrian AP collapses at high IoU).
Usage: python scripts/class_diag.py yolo26m rfdetr-medium
"""
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from kitti_errors import load_gt, iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 7: "train", 8: "truck", 9: "boat"}

gt = load_gt()
out = {}
for name in sys.argv[1:]:
    by = {}
    for d in json.load(open(f"{ROOT}/results/dets/kitti_{name}.json")):
        if d["score"] >= 0.3:
            x, y, w, h = d["bbox"]; by.setdefault(d["image_id"], []).append((d["category_id"], np.array([x, y, x + w, y + h])))
    res = {}
    for k in ["Tram", "Truck", "Van", "Pedestrian", "Cyclist"]:
        c = Counter(); n = 0
        for iid, objs in gt.items():
            dets = by.get(iid, [])
            B = np.array([b for _, b in dets]).reshape(-1, 4); C = [cc for cc, _ in dets]
            for cls, g in objs:
                if cls != k:
                    continue
                n += 1
                ov = iou(g, B)
                hit = {NAMES.get(C[j], str(C[j])) for j in np.where(ov >= 0.5)[0]}
                for hcls in hit:
                    c[hcls] += 1
                if not hit:
                    c["<none>"] += 1
        res[k] = {"n_gt": n, "frac_gt_covered_by_class": {a: round(b / n, 3) for a, b in c.most_common(6)}}
    rw, rh = [], []
    for iid, objs in gt.items():
        G = np.array([b for c, b in objs if c == "Pedestrian"]).reshape(-1, 4)
        for cc, b in by.get(iid, []):
            if cc != 1 or not len(G):
                continue
            ov = iou(b, G)
            if ov.max() >= 0.5:
                g = G[ov.argmax()]; rw.append((b[2] - b[0]) / (g[2] - g[0])); rh.append((b[3] - b[1]) / (g[3] - g[1]))
    res["pedestrian_box_ratio"] = dict(n=len(rw), w=float(np.median(rw)), h=float(np.median(rh)),
                                       w_iqr=[float(np.percentile(rw, 25)), float(np.percentile(rw, 75))])
    out[name] = res
    print(name, json.dumps(res, indent=1))
json.dump(out, open(f"{ROOT}/results/metrics/class_diag.json", "w"), indent=1)
