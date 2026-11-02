"""Print the tensor shapes flowing through YOLO26-M and RF-DETR-M (forward hooks), for a COCO-like square input
and a KITTI-shaped input. Used to draw the code-derived architecture diagrams in the report.
Usage: python scripts/inspect_shapes.py > results/shapes.txt
"""
import torch
from common import load_yolo, load_rfdetr


def shp(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [shp(i) for i in x]
    if isinstance(x, dict):
        return {k: shp(v) for k, v in x.items()}
    if hasattr(x, "tensors"):
        return shp(x.tensors)
    return type(x).__name__


@torch.inference_mode()
def yolo(h, w):
    y = load_yolo("m"); net = y.model.cuda().eval(); net.model[-1].end2end = True
    print(f"\n=== YOLO26-M, input (1,3,{h},{w}) ===")
    hooks = []
    for i, m in enumerate(net.model):
        if i in (2, 4, 6, 8, 9, 10, 16, 19, 22):
            hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: print(f"  layer {i:2d} {type(mod).__name__:7s} -> {shp(out)}")))
    head = net.model[-1]
    hooks.append(head.one2one_cv2[0].register_forward_hook(lambda m, i, o: print(f"  one2one box branch P3 -> {shp(o)}  (4*reg_max = {o.shape[1]})")))
    hooks.append(head.one2one_cv3[0].register_forward_hook(lambda m, i, o: print(f"  one2one cls branch P3 -> {shp(o)}")))
    out = net(torch.zeros(1, 3, h, w, device="cuda"))
    print(f"  Detect output (end2end, after top-K) -> {shp(out[0])}  [x1,y1,x2,y2,score,cls]")
    print(f"  anchors (grid cells) = {sum((h // s) * (w // s) for s in (8, 16, 32))}")
    for hk in hooks:
        hk.remove()


@torch.inference_mode()
def rfdetr(h, w):
    r = load_rfdetr("medium"); net = r.model.model.cuda().eval()
    print(f"\n=== RF-DETR-M, input (1,3,{h},{w}) ===")
    bb = net.backbone[0]
    hooks = [bb.encoder.register_forward_hook(lambda m, i, o: print(f"  DINOv2 windowed ViT-S encoder feats (layers 3,6,9,12) -> {shp(o)}")),
             bb.projector.register_forward_hook(lambda m, i, o: print(f"  MultiScaleProjector (C2f, P4 only) -> {shp(o)}")),
             net.transformer.register_forward_hook(lambda m, i, o: print(f"  Transformer outputs (hs, refs, enc_hs, enc_refs) -> {shp(o)}"))]
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list  # noqa
    out = net(nested_tensor_from_tensor_list([torch.zeros(3, h, w, device="cuda")]))
    print(f"  pred_logits -> {shp(out['pred_logits'])}   pred_boxes -> {shp(out['pred_boxes'])}  (cx,cy,w,h normalised)")
    print(f"  queries = {net.num_queries}, decoder layers = {net.transformer.dec_layers}, aux outputs = {len(out.get('aux_outputs', []))}")
    print(f"  tokens at stride 16 = {(h // 16) * (w // 16)}")
    for hk in hooks:
        hk.remove()


if __name__ == "__main__":
    yolo(640, 640); yolo(224, 640)
    rfdetr(576, 576); rfdetr(384, 1248)
