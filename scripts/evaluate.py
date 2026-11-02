"""Score detection dumps from predict.py with pycocotools (same evaluator for both detector families).

  python scripts/evaluate.py coco   results/dets/coco_*.json
  python scripts/evaluate.py kitti  results/dets/kitti_*.json

COCO : standard COCOeval bbox on val2017 + per-class AP for all 80 classes.
KITTI: (a) COCO-style AP over all labelled objects of 5 mapped classes (see make_kitti_gt.py);
       (b) KITTI-style AP_R40 per difficulty (Easy/Moderate/Hard) with class IoU thresholds
           Car 0.7, others 0.5; GT outside the difficulty -> ignored, detections shorter than the
           difficulty's min height -> discarded (devkit behaviour). This is a re-implementation on top
           of pycocotools, not the official C++ devkit, so values are close but not identical.
Writes results/metrics/<dataset>_<model>.json
"""
import contextlib, io, json, os, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFF = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}  # min_h, max_occ, max_trunc
KITTI_CLASS_IOU = {1: 0.5, 3: 0.7, 7: 0.5, 8: 0.5, 1001: 0.5}
R40 = np.linspace(1 / 40, 1, 40)


class KittiEval(COCOeval):
    """COCOeval + per-GT extra_ignore flag (normal IoU) + optional KITTI difficulty filter."""

    def __init__(self, gt, dt, difficulty=None):
        super().__init__(gt, dt, "bbox")
        self.difficulty = difficulty

    def _prepare(self):
        super()._prepare()
        for gts in self._gts.values():
            for g in gts:
                ig = bool(g["ignore"] or g.get("extra_ignore", 0))
                if self.difficulty and not ig:
                    mh, mo, mt = DIFF[self.difficulty]
                    if g["height"] < mh or g["occluded"] > mo or g["truncated"] > mt:
                        ig = True
                g["ignore"] = int(ig)


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def per_class(E, cat_ids):
    P = E.eval["precision"]  # [T, R, K, A, M]
    out = {}
    for k, cid in enumerate(E.params.catIds):
        if cid not in cat_ids:
            continue
        def ap(t=None, a=0):
            p = P[:, :, k, a, -1] if t is None else P[t, :, k, a, -1]
            p = p[p > -1]
            return float(p.mean()) if p.size else float("nan")
        out[cid] = dict(AP=ap(), AP50=ap(0), AP75=ap(5), APs=ap(None, 1), APm=ap(None, 2), APl=ap(None, 3))
    return out


STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def coco_eval(gt, dets):
    dt = quiet(gt.loadRes, dets)
    E = COCOeval(gt, dt, "bbox")
    quiet(E.evaluate); quiet(E.accumulate); quiet(E.summarize)
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    pc = per_class(E, set(names))
    return dict(overall=dict(zip(STAT_NAMES, map(float, E.stats))),
                per_class={names[c]: v for c, v in pc.items()})


def cyclist_composite(dets_img, min_score=0.01):
    """Build KITTI 'Cyclist' boxes from COCO person + bicycle detections in one image.

    A person p and bicycle b are paired when they overlap substantially (intersection / smaller box
    >= 0.3), overlap horizontally (x-overlap / narrower width >= 0.5) and the person's centre is above
    the bicycle's centre (rider on top). Pairs are matched greedily 1:1 by sqrt(s_p * s_b);
    output box = union of the two boxes, score = sqrt(s_p * s_b).
    """
    P = [d for d in dets_img if d["category_id"] == 1 and d["score"] >= min_score]
    B = [d for d in dets_img if d["category_id"] == 2 and d["score"] >= min_score]
    if not P or not B:
        return []
    pb = np.array([d["bbox"] for d in P]); bb = np.array([d["bbox"] for d in B])
    ps = np.array([d["score"] for d in P]); bs = np.array([d["score"] for d in B])
    px1, py1, px2, py2 = pb[:, 0], pb[:, 1], pb[:, 0] + pb[:, 2], pb[:, 1] + pb[:, 3]
    bx1, by1, bx2, by2 = bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]
    ix = np.clip(np.minimum(px2[:, None], bx2[None]) - np.maximum(px1[:, None], bx1[None]), 0, None)
    iy = np.clip(np.minimum(py2[:, None], by2[None]) - np.maximum(py1[:, None], by1[None]), 0, None)
    inter = ix * iy
    amin = np.minimum((pb[:, 2] * pb[:, 3])[:, None], (bb[:, 2] * bb[:, 3])[None])
    wmin = np.minimum(pb[:, 2][:, None], bb[:, 2][None])
    above = ((py1 + py2) / 2)[:, None] < ((by1 + by2) / 2)[None]
    ok = (inter / np.maximum(amin, 1e-6) >= 0.3) & (ix / np.maximum(wmin, 1e-6) >= 0.5) & above
    score = np.sqrt(ps[:, None] * bs[None]) * ok
    out, used_p, used_b = [], set(), set()
    for flat in np.argsort(-score, axis=None):
        i, j = np.unravel_index(flat, score.shape)
        if score[i, j] <= 0:
            break
        if i in used_p or j in used_b:
            continue
        used_p.add(i); used_b.add(j)
        x1, y1 = min(px1[i], bx1[j]), min(py1[i], by1[j]); x2, y2 = max(px2[i], bx2[j]), max(py2[i], by2[j])
        out.append(dict(image_id=P[0]["image_id"], category_id=1001, bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        score=float(score[i, j])))
    return out


def kitti_preds(dets):
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out = []
    for iid, ds in by_img.items():
        out += [d for d in ds if d["category_id"] in (1, 3, 7, 8)]
        out += cyclist_composite(ds)
    return out


def kitti_eval(gt, dets):
    preds = kitti_preds(dets)
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    res = {}
    # (a) COCO-style over all labelled objects
    dt = quiet(gt.loadRes, preds)
    E = KittiEval(gt, dt)
    quiet(E.evaluate); quiet(E.accumulate); quiet(E.summarize)
    pc = per_class(E, set(names))
    res["coco_style"] = dict(overall_5cls=dict(zip(STAT_NAMES, map(float, E.stats))),
                             per_class={names[c]: v for c, v in pc.items()})
    core = [pc[c]["AP"] for c in (3, 1, 1001)]
    res["coco_style"]["mAP_car_ped_cyc"] = float(np.mean(core))
    # (b) KITTI-style difficulty AP_R40
    res["kitti_R40"] = {}
    for diff, (min_h, _, _) in DIFF.items():
        p = [d for d in preds if d["bbox"][3] >= min_h]
        dt = quiet(gt.loadRes, p)
        E = KittiEval(gt, dt, difficulty=diff)
        E.params.iouThrs = np.array([0.5, 0.7]); E.params.recThrs = R40
        E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]; E.params.maxDets = [100]
        quiet(E.evaluate); quiet(E.accumulate)
        P = E.eval["precision"]
        d = {}
        for k, cid in enumerate(E.params.catIds):
            t = 1 if KITTI_CLASS_IOU[cid] == 0.7 else 0
            pr = P[t, :, k, 0, 0]
            d[names[cid]] = float(pr.mean()) if (pr > -1).all() else float("nan")
            # also report Car at IoU 0.5 for a localisation-tolerant view
            if cid == 3:
                d["Car@0.5"] = float(P[0, :, k, 0, 0].mean())
        res["kitti_R40"][diff] = d
    return res


def main():
    dataset, files = sys.argv[1], sys.argv[2:]
    gt_path = (os.path.join(ROOT, "data/coco/annotations/instances_val2017.json") if dataset == "coco"
               else os.path.join(ROOT, "data/kitti/kitti_train_cocofmt.json"))
    gt = quiet(COCO, gt_path)
    os.makedirs(os.path.join(ROOT, "results/metrics"), exist_ok=True)
    for f in files:
        if f.endswith(".meta.json"):
            continue
        name = os.path.basename(f)[len(dataset) + 1:-5]
        dets = json.load(open(f))
        r = coco_eval(gt, dets) if dataset == "coco" else kitti_eval(gt, dets)
        meta_f = f[:-5] + ".meta.json"
        if os.path.exists(meta_f):
            r["meta"] = json.load(open(meta_f))
        json.dump(r, open(os.path.join(ROOT, f"results/metrics/{dataset}_{name}.json"), "w"), indent=1)
        if dataset == "coco":
            o = r["overall"]
            print(f"{name:22s} AP {o['AP']*100:5.1f} AP50 {o['AP50']*100:5.1f} AP75 {o['AP75']*100:5.1f} "
                  f"APs {o['APs']*100:5.1f} APm {o['APm']*100:5.1f} APl {o['APl']*100:5.1f} AR100 {o['AR100']*100:5.1f}")
        else:
            c = r["coco_style"]; m = r["kitti_R40"]["moderate"]
            print(f"{name:22s} mAP(C/P/Cy) {c['mAP_car_ped_cyc']*100:5.1f} | " +
                  " ".join(f"{k} {v['AP']*100:5.1f}" for k, v in c["per_class"].items()) +
                  " | Mod R40: " + " ".join(f"{k} {v*100:5.1f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
