"""COCO AP as a function of scene crowdedness - probes NMS vs NMS-free (one-to-one / set prediction) behaviour.

Images are binned by number of non-crowd GT instances; COCOeval is run on each subset. Also reports
person-class AP on images with >= 10 people, where heavily overlapping same-class objects are what a
fixed-IoU NMS can wrongly suppress.
Usage: python scripts/crowd_analysis.py results/dets/coco_*.json
"""
import contextlib, io, json, os, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = [(1, 5), (6, 15), (16, 10**6)]


def q(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def ap(gt, dt, img_ids, cat_ids=None):
    E = COCOeval(gt, dt, "bbox"); E.params.imgIds = img_ids
    if cat_ids:
        E.params.catIds = cat_ids
    q(E.evaluate); q(E.accumulate); q(E.summarize)
    return float(E.stats[0]), float(E.stats[1]), float(E.stats[8])


def main():
    gt = q(COCO, os.path.join(ROOT, "data/coco/annotations/instances_val2017.json"))
    n_obj = {i: len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in gt.getImgIds()}
    n_person = {i: len(gt.getAnnIds(imgIds=i, catIds=[1], iscrowd=False)) for i in gt.getImgIds()}
    subsets = {f"{lo}-{hi if hi < 10**6 else '+'} objs": [i for i, n in n_obj.items() if lo <= n <= hi] for lo, hi in BINS}
    crowd_people = [i for i, n in n_person.items() if n >= 10]
    out = {"_sizes": {k: len(v) for k, v in subsets.items()} | {"person>=10": len(crowd_people)}}
    for f in sys.argv[1:]:
        if f.endswith(".meta.json"):
            continue
        name = os.path.basename(f)[5:-5]
        dt = q(gt.loadRes, json.load(open(f)))
        r = {k: ap(gt, dt, v)[0] for k, v in subsets.items()}
        r["person AP (>=10 people)"], _, r["person AR100 (>=10 people)"] = ap(gt, dt, crowd_people, [1])
        out[name] = r
        print(name, {k: round(v * 100, 1) for k, v in r.items()})
    json.dump(out, open(os.path.join(ROOT, "results/metrics/coco_crowd.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
