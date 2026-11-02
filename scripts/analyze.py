"""Aggregate metrics into markdown tables (results/tables.md) and figures (report/figs/*.png)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "results/metrics")
FIG = os.path.join(ROOT, "report/figs")
# validated categorical slots (dataviz reference palette, light mode), fixed order per entity
C_YOLO, C_RF, C_NMS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

YOLO = ["n", "s", "m", "l", "x"]
RF = ["nano", "small", "medium", "large"]
PAIRS = [("small", "yolo26s", "rfdetr-nano"), ("medium", "yolo26m", "rfdetr-medium"), ("large", "yolo26l", "rfdetr-large")]
PUB = {"yolo26n": 40.1, "yolo26s": 47.8, "yolo26m": 52.5, "yolo26l": 54.4, "yolo26x": 56.9,
       "yolo26n-nms": 40.9, "yolo26s-nms": 48.6, "yolo26m-nms": 53.1, "yolo26l-nms": 55.0, "yolo26x-nms": 57.5,
       "rfdetr-nano": 48.4, "rfdetr-small": 53.0, "rfdetr-medium": 54.7, "rfdetr-large": 56.5}
LABEL = {**{f"yolo26{s}": f"YOLO26-{s.upper()}" for s in YOLO}, **{f"yolo26{s}-nms": f"YOLO26-{s.upper()} (NMS head)" for s in YOLO},
         **{f"rfdetr-{s}": f"RF-DETR-{s[0].upper()}" for s in RF}}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def lat():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/latency.jsonl"))]
    d = {}
    for r in rows:
        key = ("yolo26" + r["size"] + ("-nms" if r["model"].endswith("-nms") else "")) if r["family"] == "yolo" else "rfdetr-" + r["size"]
        d[(key, r["batch"], r.get("tag", ""), tuple(r["input_hw"]))] = r
    return d


def L(d, key, b=1, tag=""):
    for k, v in d.items():
        if k[0] == key and k[1] == b and k[2] == tag:
            return v
    return None


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main():
    out = []
    lt = lat()
    coco = {os.path.basename(p)[5:-5]: load(p) for p in glob.glob(f"{M}/coco_*.json") if "crowd" not in p}
    kitti = {os.path.basename(p)[6:-5]: load(p) for p in glob.glob(f"{M}/kitti_*.json") if "errors" not in p}

    # ---------------- latency table ----------------
    out.append("## Latency / memory (RTX 5070, FP16, PyTorch eager, network forward incl. on-GPU top-K/NMS)\n")
    out.append("| Model | Params (M) | Input | Batch-1 latency ms (median) | Batch-16 ms / image | Peak alloc. MB (b=1) | Peak alloc. MB (b=16) | COCO end-to-end pipeline ms (median / p90) |")
    out.append("|---|---|---|---|---|---|---|---|")
    order = [f"yolo26{s}" for s in YOLO] + [f"yolo26{s}-nms" for s in YOLO] + [f"rfdetr-{s}" for s in RF]
    for k in order:
        a, b = L(lt, k, 1), L(lt, k, 16)
        if not a:
            continue
        m = coco.get(k, {}).get("meta") if coco.get(k) else None
        e2e = f"{m['e2e_median_ms']:.2f} / {m['e2e_p90_ms']:.2f}" if m else "–"
        out.append(f"| {LABEL[k]} | {a['params_M']:.1f} | {a['input_hw'][0]}×{a['input_hw'][1]} | {a['median_ms']:.2f} | "
                   f"{b['per_image_ms']:.2f} | {a['peak_mem_MB']:.0f} | {b['peak_mem_MB']:.0f} | {e2e} |")
    kt = [(k, v) for k, v in lt.items() if k[2] == "kitti"]
    if kt:
        out.append("\nKITTI-shaped inputs (batch 1):\n\n| Model | Input H×W | Batch-1 latency ms |\n|---|---|---|")
        for k, v in sorted(kt):
            out.append(f"| {LABEL[k[0]]} | {k[3][0]}×{k[3][1]} | {v['median_ms']:.2f} |")

    # ---------------- COCO table ----------------
    out.append("\n## COCO val2017 (5000 images), pycocotools bbox\n")
    out.append("| Model | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR100 | Published AP | Δ vs published |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        if k not in coco:
            continue
        o = coco[k]["overall"]
        out.append(f"| {LABEL[k]} | **{pct(o['AP'])}** | {pct(o['AP50'])} | {pct(o['AP75'])} | {pct(o['APs'])} | {pct(o['APm'])} | "
                   f"{pct(o['APl'])} | {pct(o['AR1'])} | {pct(o['AR100'])} | {PUB[k]:.1f} | {100 * o['AP'] - PUB[k]:+.1f} |")

    # ---------------- matched pairs ----------------
    out.append("\n## Matched-compute pairs (YOLO26 e2e head vs RF-DETR)\n")
    out.append("| Tier | Model | B1 ms | B16 ms/img | COCO AP | AP50 | AP75 | APs | APl | KITTI mAP (Car/Ped/Cyc) | KITTI Car AP | KITTI Ped AP | KITTI Cyc AP | Car R40 mod. (IoU .7) | Ped R40 mod. |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tier, y, r in PAIRS:
        for k in (y, r):
            a, b = L(lt, k, 1), L(lt, k, 16)
            c = coco.get(k, {}).get("overall", {}); kk = kitti.get(k)
            kc = kk["coco_style"] if kk else None
            km = kk["kitti_R40"]["moderate"] if kk else {}
            out.append(f"| {tier} | {LABEL[k]} | {a['median_ms']:.2f} | {b['per_image_ms']:.2f} | {pct(c.get('AP'))} | {pct(c.get('AP50'))} | "
                       f"{pct(c.get('AP75'))} | {pct(c.get('APs'))} | {pct(c.get('APl'))} | "
                       f"{pct(kc['mAP_car_ped_cyc']) if kc else '–'} | {pct(kc['per_class']['Car']['AP']) if kc else '–'} | "
                       f"{pct(kc['per_class']['Pedestrian']['AP']) if kc else '–'} | {pct(kc['per_class']['Cyclist']['AP']) if kc else '–'} | "
                       f"{pct(km.get('Car'))} | {pct(km.get('Pedestrian'))} |")

    # ---------------- KITTI tables ----------------
    korder = [k for k in order + sorted(set(kitti) - set(order)) if k in kitti]
    out.append("\n## KITTI 2D (training split, 7481 images, zero-shot) — COCO-style AP over all labelled objects\n")
    out.append("| Model | input | pipeline ms | mAP C/P/Cy | Car AP | Car AP50 | Car AP75 | Ped AP | Ped AP50 | Cyc AP | Cyc AP50 | Truck AP | Tram AP | APs (5cls) | APm | APl | AR100 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in korder:
        c = kitti[k]["coco_style"]; p = c["per_class"]; o = c["overall_5cls"]; m = kitti[k].get("meta", {})
        out.append(f"| {LABEL.get(k, k)} | {m.get('imgsz')} | {m.get('e2e_median_ms', float('nan')):.2f} | **{pct(c['mAP_car_ped_cyc'])}** | {pct(p['Car']['AP'])} | {pct(p['Car']['AP50'])} | "
                   f"{pct(p['Car']['AP75'])} | {pct(p['Pedestrian']['AP'])} | {pct(p['Pedestrian']['AP50'])} | {pct(p['Cyclist']['AP'])} | "
                   f"{pct(p['Cyclist']['AP50'])} | {pct(p['Truck']['AP'])} | {pct(p['Tram']['AP'])} | {pct(o['APs'])} | {pct(o['APm'])} | {pct(o['APl'])} | {pct(o['AR100'])} |")
    out.append("\n## KITTI-style AP_R40 (Car @ IoU 0.7, Ped/Cyc @ 0.5; Car@0.5 shown for reference)\n")
    out.append("| Model | Car E | Car M | Car H | Car@0.5 M | Ped E | Ped M | Ped H | Cyc E | Cyc M | Cyc H |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in korder:
        R = kitti[k]["kitti_R40"]
        out.append(f"| {LABEL.get(k, k)} | " + " | ".join(
            [pct(R[d]["Car"]) for d in ("easy", "moderate", "hard")] + [pct(R["moderate"]["Car@0.5"])] +
            [pct(R[d][c]) for c in ("Pedestrian", "Cyclist") for d in ("easy", "moderate", "hard")]) + " |")

    # ---------------- cross-dataset per-class ----------------
    out.append("\n## Cross-dataset: same COCO class, COCO val AP → KITTI AP (absolute drop in AP points)\n")
    pairs_cls = [("person", "Pedestrian"), ("car", "Car"), ("truck", "Truck"), ("train", "Tram"), ("bicycle", "Cyclist")]
    out.append("| Model | " + " | ".join(f"{a}→{b}" for a, b in pairs_cls) + " | APs | APm | APl |")
    out.append("|---|" + "---|" * (len(pairs_cls) + 3))
    for k in order:
        if k in coco and k in kitti:
            cells = []
            for a, b in pairs_cls:
                ca = coco[k]["per_class"][a]["AP"]; kb = kitti[k]["coco_style"]["per_class"][b]["AP"]
                cells.append(f"{pct(ca)}→{pct(kb)} ({100 * (kb - ca):+.1f})")
            co, ko = coco[k]["overall"], kitti[k]["coco_style"]["overall_5cls"]
            for s in ("APs", "APm", "APl"):
                cells.append(f"{pct(co[s])}→{pct(ko[s])}")
            out.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")

    # ---------------- crowd ----------------
    cr = load(f"{M}/coco_crowd.json")
    if cr:
        sizes = cr.pop("_sizes")
        keys = list(sizes)
        out.append("\n## COCO AP vs scene crowdedness (NMS vs NMS-free)\n")
        out.append("| Model | " + " | ".join(f"{k} (n={sizes[k]})" for k in keys) + " | person AR100 (≥10 people) |")
        out.append("|---|" + "---|" * (len(keys) + 1))
        for k in order:
            if k in cr:
                r = cr[k]
                out.append(f"| {LABEL[k]} | " + " | ".join(pct(r[x]) if x in r else pct(r.get('person AP (>=10 people)')) for x in keys) +
                           f" | {pct(r['person AR100 (>=10 people)'])} |")

    # ---------------- errors ----------------
    errs = {os.path.basename(p)[13:-5]: load(p) for p in glob.glob(f"{M}/kitti_errors_*.json")}
    if errs:
        out.append("\n## KITTI error breakdown at each model's F1-optimal threshold (IoU 0.5)\n")
        out.append("| Model | Class | thr | P | R | F1 | TP | dup | loc | confusion | background | ignored | max recall <25px | 25-40 | 40-80 | 80-160 | >160 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in [x for x in order + sorted(set(errs) - set(order)) if x in errs]:
            for c, v in errs[k].items():
                e = v["errors"]; h = v["max_recall_by_height"]
                out.append(f"| {LABEL.get(k, k)} | {c} | {v['thr']:.2f} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {e['TP']} | "
                           f"{e['duplicate']} | {e['loc']} | {e['confusion']} | {e['background']} | {e['ignored']} | " +
                           " | ".join(pct(h[b]) for b in ["<25", "25-40", "40-80", "80-160", ">160"]) + " |")
        n = next(iter(errs.values()))
        out.append("\nGT counts by height: " + "; ".join(f"{c}: {v['n_by_height']}" for c, v in n.items()))

    open(os.path.join(ROOT, "results/tables.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    figures(lt, coco, kitti, errs)


def scatter_family(ax, pts, color, label, marker="o", ls="-"):
    pts = [p for p in pts if p[1] is not None]
    if not pts:
        return
    xs, ys, names = zip(*pts)
    ax.plot(xs, ys, ls, color=color, lw=2, zorder=2)
    ax.scatter(xs, ys, s=64, color=color, edgecolor=SURF, linewidth=2, zorder=3, label=label, marker=marker)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=INK2)


def figures(lt, coco, kitti, errs):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF})
    # 1. speed-accuracy on COCO, two panels (batch-1 latency, batch-16 throughput)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, b, xl in [(axes[0], 1, "Batch-1 latency (ms, median)"), (axes[1], 16, "Batch-16 time per image (ms)")]:
        style(ax)
        y = [(L(lt, f"yolo26{s}", b)["per_image_ms"] if b > 1 else L(lt, f"yolo26{s}", b)["median_ms"], 100 * coco[f"yolo26{s}"]["overall"]["AP"] if f"yolo26{s}" in coco else None, s.upper()) for s in YOLO]
        yn = [(L(lt, f"yolo26{s}-nms", b)["per_image_ms"] if b > 1 else L(lt, f"yolo26{s}-nms", b)["median_ms"], 100 * coco[f"yolo26{s}-nms"]["overall"]["AP"] if f"yolo26{s}-nms" in coco else None, "") for s in YOLO]
        r = [(L(lt, f"rfdetr-{s}", b)["per_image_ms"] if b > 1 else L(lt, f"rfdetr-{s}", b)["median_ms"], 100 * coco[f"rfdetr-{s}"]["overall"]["AP"] if f"rfdetr-{s}" in coco else None, s[0].upper()) for s in RF]
        scatter_family(ax, y, C_YOLO, "YOLO26")
        scatter_family(ax, r, C_RF, "RF-DETR")
        ax.set_xlabel(xl, color=INK); ax.set_ylabel("COCO val2017 AP (%)", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.suptitle("Speed vs. accuracy on COCO (RTX 5070, FP16)", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/coco_speed_accuracy.png", dpi=160); plt.close(fig)

    # 2. COCO vs KITTI per tier (grouped bars)
    if all(k in kitti for _, y, r in PAIRS for k in (y, r)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        metrics = [("COCO AP", lambda k: coco[k]["overall"]["AP"]), ("KITTI Car AP", lambda k: kitti[k]["coco_style"]["per_class"]["Car"]["AP"]),
                   ("KITTI Ped AP", lambda k: kitti[k]["coco_style"]["per_class"]["Pedestrian"]["AP"]),
                   ("KITTI Cyc AP", lambda k: kitti[k]["coco_style"]["per_class"]["Cyclist"]["AP"])]
        x = np.arange(len(metrics)); w = 0.38
        for ax, (tier, y, r) in zip(axes, PAIRS):
            style(ax); ax.grid(axis="x", visible=False)
            for off, k, col in [(-w / 2 - 0.01, y, C_YOLO), (w / 2 + 0.01, r, C_RF)]:
                vals = [100 * f(k) for _, f in metrics]
                bars = ax.bar(x + off, vals, w, color=col, label=LABEL[k])
                for bx, v in zip(x + off, vals):
                    ax.text(bx, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
            ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metrics], fontsize=8.5, color=INK2)
            ax.set_title(f"{tier} tier", color=INK, fontsize=10.5)
            ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
        axes[0].set_ylabel("AP[.5:.95] (%)", color=INK)
        fig.tight_layout(); fig.savefig(f"{FIG}/coco_vs_kitti_pairs.png", dpi=160); plt.close(fig)

    # 3. KITTI max recall by object height, matched medium pair
    if errs and "yolo26m" in errs and "rfdetr-medium" in errs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        bins = ["<25", "25-40", "40-80", "80-160", ">160"]
        series = [("yolo26m", C_YOLO, "o", "-"), ("rfdetr-medium", C_RF, "o", "-")]
        for ax, c in zip(axes, ["Car", "Pedestrian", "Cyclist"]):
            style(ax)
            for k, col, mk, ls in series:
                if k not in errs:
                    continue
                v = [100 * (errs[k][c]["max_recall_by_height"][b] or 0) for b in bins]
                ax.plot(bins, v, ls, color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=1.5, label=LABEL.get(k, k.replace("yolo26m", "YOLO26-M").replace("rfdetr-medium", "RF-DETR-M")))
            ax.set_title(c, color=INK, fontsize=10.5); ax.set_xlabel("GT box height (px)", color=INK)
        axes[0].set_ylabel("Max recall @ IoU 0.5 (%)", color=INK)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
        fig.tight_layout(); fig.savefig(f"{FIG}/kitti_recall_by_height.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
