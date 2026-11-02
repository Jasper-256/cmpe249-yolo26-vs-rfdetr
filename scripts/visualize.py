"""Qualitative KITTI comparison of two models at their F1-optimal thresholds (from kitti_errors.py).

Selects images where the two models disagree most in #TP for Car/Pedestrian/Cyclist (both directions)
plus one fixed crowded-pedestrian scene, and draws GT (white dashed) + detections (TP solid, FP red).
Usage: python scripts/visualize.py yolo26m rfdetr-medium
"""
import json, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import kitti_preds
from kitti_errors import load_gt, iou, CLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"Car": (214, 120, 42), "Pedestrian": (122, 175, 27), "Cyclist": (0, 161, 237)}  # BGR: blue, teal, amber; FP = red


def per_image(name):
    preds = kitti_preds(json.load(open(f"{ROOT}/results/dets/kitti_{name}.json")))
    thr = {c: v["thr"] for c, v in json.load(open(f"{ROOT}/results/metrics/kitti_errors_{name}.json")).items()}
    inv = {v: k for k, v in CLS.items()}
    out = {}
    for d in preds:
        c = inv.get(d["category_id"])
        if c and d["score"] >= thr[c]:
            x, y, w, h = d["bbox"]
            out.setdefault(d["image_id"], []).append((c, np.array([x, y, x + w, y + h]), d["score"]))
    return out, thr


def tp_flags(dets, objs):
    flags, used = [], set()
    for c, b, s in sorted(dets, key=lambda t: -t[2]):
        gts = [(i, g) for i, (gc, g) in enumerate(objs) if gc == c and i not in used]
        ok = False
        if gts:
            ov = iou(b, np.array([g for _, g in gts]))
            if ov.max() >= 0.5:
                used.add(gts[int(ov.argmax())][0]); ok = True
        flags.append((c, b, s, ok))
    return flags


def draw(img, objs, flags, title):
    im = img.copy()
    for c, g in objs:
        if c in CLS:
            x1, y1, x2, y2 = g.astype(int)
            for xx in range(x1, x2, 10):
                cv2.line(im, (xx, y1), (min(xx + 5, x2), y1), (255, 255, 255), 1); cv2.line(im, (xx, y2), (min(xx + 5, x2), y2), (255, 255, 255), 1)
            for yy in range(y1, y2, 10):
                cv2.line(im, (x1, yy), (x1, min(yy + 5, y2)), (255, 255, 255), 1); cv2.line(im, (x2, yy), (x2, min(yy + 5, y2)), (255, 255, 255), 1)
    for c, b, s, ok in flags:
        x1, y1, x2, y2 = b.astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), COL[c] if ok else (40, 40, 230), 2)
        cv2.putText(im, f"{c[:3]} {s:.2f}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL[c] if ok else (40, 40, 230), 1, cv2.LINE_AA)
    bar = np.full((22, im.shape[1], 3), 252, np.uint8)
    cv2.putText(bar, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def main():
    a, b = sys.argv[1], sys.argv[2]
    gt = load_gt()
    pa, ta = per_image(a); pb, tb = per_image(b)
    score = {}
    for iid, objs in gt.items():
        fa = sum(f[3] for f in tp_flags(pa.get(iid, []), objs)); fb = sum(f[3] for f in tp_flags(pb.get(iid, []), objs))
        score[iid] = fa - fb
    ranked = sorted(score, key=lambda i: score[i])
    n_ped = {i: sum(c == "Pedestrian" for c, _ in o) for i, o in gt.items()}
    picks = [ranked[-1], ranked[-2], ranked[0], ranked[1], max(n_ped, key=n_ped.get)]
    os.makedirs(f"{ROOT}/report/figs", exist_ok=True)
    for k, iid in enumerate(picks):
        img = cv2.imread(f"{ROOT}/data/kitti/training/image_2/{iid:06d}.png")
        objs = gt[iid]
        fa, fb = tp_flags(pa.get(iid, []), objs), tp_flags(pb.get(iid, []), objs)
        panel = np.vstack([draw(img, objs, fa, f"{a}  img {iid:06d}  TP={sum(f[3] for f in fa)} FP={sum(not f[3] for f in fa)}"),
                           draw(img, objs, fb, f"{b}  img {iid:06d}  TP={sum(f[3] for f in fb)} FP={sum(not f[3] for f in fb)}")])
        cv2.imwrite(f"{ROOT}/report/figs/qual_{k}_{iid:06d}.png", panel)
        print(f"qual_{k}_{iid:06d}.png  TP diff {score[iid]:+d}")


if __name__ == "__main__":
    main()
