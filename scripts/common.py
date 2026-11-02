"""Shared model loading helpers for YOLO26 and RF-DETR experiments."""
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")

YOLO_SIZES = ["n", "s", "m", "l", "x"]
RFDETR_SIZES = ["nano", "small", "medium", "large"]


def load_yolo(size: str):
    from ultralytics import YOLO
    return YOLO(os.path.join(WEIGHTS, f"yolo26{size}.pt"))


def load_rfdetr(size: str, resolution: int | None = None):
    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
           "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[size]
    kw = {}
    if resolution is not None:
        kw["resolution"] = resolution
    cwd = os.getcwd()
    os.chdir(WEIGHTS)  # rfdetr downloads checkpoints into the cwd
    try:
        m = cls(**kw)
    finally:
        os.chdir(cwd)
    return m


def gpu_info():
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
