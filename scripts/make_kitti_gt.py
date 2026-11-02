"""Convert KITTI 2D object labels (training split, 7481 imgs) into a COCO-format GT file whose
category ids line up with COCO-91 ids so COCO-pretrained detectors can be scored without retraining.

Taxonomy mapping (KITTI eval class <- COCO prediction class), with KITTI "neighbour" classes that are
ignored (neither TP nor FP, like the official KITTI devkit does for Van/Car and Person_sitting/Pedestrian):

  eval class  | id   | KITTI GT       | ignored GT (normal IoU)            | COCO prediction source
  Car         | 3    | Car            | Van                                | car
  Pedestrian  | 1    | Pedestrian     | Person_sitting, Cyclist (riders)   | person
  Cyclist     | 1001 | Cyclist        | -                                  | person+bicycle composite (see evaluate.py)
  Truck       | 8    | Truck          | Van, Misc                          | truck
  Tram        | 7    | Tram           | -                                  | train
DontCare regions are added to every category as iscrowd=1 (pycocotools then uses intersection/det-area,
so detections inside DontCare are ignored, mirroring the devkit). Misc and Van are never scored as classes.

Each GT also stores KITTI height/occlusion/truncation so evaluate.py can apply Easy/Moderate/Hard filters.
"""
import glob, json, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KD = os.path.join(ROOT, "data/kitti/training")

CATS = [dict(id=1, name="Pedestrian"), dict(id=3, name="Car"), dict(id=7, name="Tram"),
        dict(id=8, name="Truck"), dict(id=1001, name="Cyclist")]
VALID = {"Car": 3, "Pedestrian": 1, "Cyclist": 1001, "Truck": 8, "Tram": 7}
NEIGHBOUR_IGNORE = {"Van": [3, 8], "Person_sitting": [1], "Cyclist": [1], "Misc": [8]}


def main():
    images, anns = [], []
    aid = 1
    for lf in sorted(glob.glob(os.path.join(KD, "label_2/*.txt"))):
        stem = os.path.basename(lf)[:-4]
        iid = int(stem)
        w, h = Image.open(os.path.join(KD, "image_2", stem + ".png")).size
        images.append(dict(id=iid, file_name=stem + ".png", width=w, height=h))
        for line in open(lf):
            f = line.split()
            cls, trunc, occ = f[0], float(f[1]), int(f[2])
            x1, y1, x2, y2 = map(float, f[4:8])
            bw, bh = x2 - x1, y2 - y1
            base = dict(image_id=iid, bbox=[x1, y1, bw, bh], area=bw * bh, truncated=trunc, occluded=occ,
                        height=bh, kitti_class=cls)
            targets = []
            if cls in VALID:
                targets.append((VALID[cls], 0, 0))
            for cid in NEIGHBOUR_IGNORE.get(cls, []):
                targets.append((cid, 0, 1))
            if cls == "DontCare":
                targets += [(c["id"], 1, 1) for c in CATS]
            for cid, crowd, ign in targets:
                anns.append(dict(base, id=aid, category_id=cid, iscrowd=crowd, extra_ignore=ign))
                aid += 1
    out = dict(images=images, annotations=anns, categories=CATS)
    json.dump(out, open(os.path.join(ROOT, "data/kitti/kitti_train_cocofmt.json"), "w"))
    print(len(images), "images", len(anns), "annotations")


if __name__ == "__main__":
    main()
