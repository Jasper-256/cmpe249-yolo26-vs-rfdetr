"""KITTI failure analysis for zero-shot COCO detectors.

For each model and class (Car, Pedestrian, Cyclist):
  * recall@IoU0.5 by GT box height bin, using all detections (score >= 0.001, top-300) -> "max recall"
  * the score threshold that maximises F1 (IoU 0.5) -> operating point; at that threshold every detection is
    classified greedily (highest score first) as
      TP          IoU >= 0.5 with an unmatched valid GT of the class
      duplicate   IoU >= 0.5 with an already-matched GT of the class
      ignored     IoU >= 0.5 with a neighbour/ignored GT, or >= 50% inside a DontCare region
      loc         0.1 <= best IoU < 0.5 with a valid GT of the class (mis-localised)
      confusion   IoU >= 0.5 with a GT of another KITTI class (e.g. car box on a Van/Truck/Misc)
      background  everything else (no labelled object there)
    and missed GTs are counted by height bin.
Usage: python scripts/kitti_errors.py results/dets/kitti_<model>.json ...
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import kitti_preds  # same COCO->KITTI prediction mapping (incl. cyclist composite)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = {"Car": 3, "Pedestrian": 1, "Cyclist": 1001}
IGN = {"Car": {"Van"}, "Pedestrian": {"Person_sitting", "Cyclist"}, "Cyclist": set()}
HBINS = [0, 25, 40, 80, 160, 1e9]
HLAB = ["<25", "25-40", "40-80", "80-160", ">160"]


def load_gt():
    gt = {}
    for lf in sorted(glob.glob(os.path.join(ROOT, "data/kitti/training/label_2/*.txt"))):
        iid = int(os.path.basename(lf)[:-4])
        objs = []
        for line in open(lf):
            f = line.split()
            objs.append((f[0], np.array(list(map(float, f[4:8])))))
        gt[iid] = objs
    return gt


def iou(a, B):
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def ioa(a, B):  # intersection / area(a)
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def match_class(cname, preds_by_img, gt, thr):
    """Greedy matching at score >= thr. Returns error counts, per-GT matched flags and heights."""
    cid = CLS[cname]
    counts = dict(TP=0, duplicate=0, ignored=0, loc=0, confusion=0, background=0)
    gt_h, gt_hit = [], []
    for iid, objs in gt.items():
        valid = np.array([b for c, b in objs if c == cname]).reshape(-1, 4)
        ign = np.array([b for c, b in objs if c in IGN[cname]]).reshape(-1, 4)
        dc = np.array([b for c, b in objs if c == "DontCare"]).reshape(-1, 4)
        other = np.array([b for c, b in objs if c not in IGN[cname] | {cname, "DontCare"}]).reshape(-1, 4)
        hit = np.zeros(len(valid), bool)
        for d in preds_by_img.get(iid, {}).get(cid, []):
            if d[4] < thr:
                break
            box = d[:4]
            ov = iou(box, valid)
            if len(ov) and ov.max() >= 0.5:
                cand = np.where((ov >= 0.5) & ~hit)[0]
                if len(cand):
                    hit[cand[np.argmax(ov[cand])]] = True; counts["TP"] += 1
                else:
                    counts["duplicate"] += 1
            elif (len(ign) and iou(box, ign).max() >= 0.5) or (len(dc) and ioa(box, dc).max() >= 0.5):
                counts["ignored"] += 1
            elif len(ov) and ov.max() >= 0.1:
                counts["loc"] += 1
            elif len(other) and iou(box, other).max() >= 0.5:
                counts["confusion"] += 1
            else:
                counts["background"] += 1
        gt_h += list(valid[:, 3] - valid[:, 1]); gt_hit += list(hit)
    return counts, np.array(gt_h), np.array(gt_hit)


def best_f1_threshold(cname, preds_by_img, gt):
    """Sweep thresholds over the score quantiles to find F1-optimal operating point (IoU 0.5)."""
    scores = np.concatenate([[d[4] for d in v.get(CLS[cname], [])] for v in preds_by_img.values()] + [[]])
    cands = np.unique(np.quantile(scores[scores >= 0.01], np.linspace(0, 0.995, 30))) if len(scores) else [0.5]
    best = (0, 0.5, None)
    for t in cands:
        c, h, hit = match_class(cname, preds_by_img, gt, t)
        p = c["TP"] / max(c["TP"] + c["duplicate"] + c["loc"] + c["confusion"] + c["background"], 1)
        r = hit.mean() if len(hit) else 0
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[0]:
            best = (f1, float(t), (c, p, r))
    return best


def main():
    gt = load_gt()
    out_all = {}
    for f in sys.argv[1:]:
        name = os.path.basename(f)[len("kitti_"):-5]
        preds = kitti_preds(json.load(open(f)))
        by_img = {}
        for d in preds:
            x, y, w, h = d["bbox"]
            by_img.setdefault(d["image_id"], {}).setdefault(d["category_id"], []).append([x, y, x + w, y + h, d["score"]])
        for v in by_img.values():
            for k in v:
                v[k] = np.array(sorted(v[k], key=lambda r: -r[4]))
        res = {}
        for cname in CLS:
            _, h_all, hit_all = match_class(cname, by_img, gt, 0.0)
            f1, thr, (c, p, r) = best_f1_threshold(cname, by_img, gt)
            _, h_op, hit_op = match_class(cname, by_img, gt, thr)
            idx = np.digitize(h_all, HBINS) - 1
            res[cname] = dict(
                n_gt=int(len(h_all)), f1=f1, thr=thr, precision=p, recall=r, errors=c,
                max_recall_by_height={HLAB[i]: float(hit_all[idx == i].mean()) if (idx == i).any() else None for i in range(len(HLAB))},
                op_recall_by_height={HLAB[i]: float(hit_op[idx == i].mean()) if (idx == i).any() else None for i in range(len(HLAB))},
                n_by_height={HLAB[i]: int((idx == i).sum()) for i in range(len(HLAB))})
            print(f"{name:24s} {cname:10s} F1 {f1:.3f} @thr {thr:.3f}  P {p:.3f} R {r:.3f}  errors {c}")
        out_all[name] = res
        os.makedirs(os.path.join(ROOT, "results/metrics"), exist_ok=True)
        json.dump(res, open(os.path.join(ROOT, f"results/metrics/kitti_errors_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
