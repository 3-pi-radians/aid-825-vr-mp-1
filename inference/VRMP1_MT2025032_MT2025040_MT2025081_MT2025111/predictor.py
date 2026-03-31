"""
predictor.py — Student inference file for hidden evaluation.

╔══════════════════════════════════════════════════════════════════╗
║  DO NOT RENAME ANY FUNCTION.                                    ║
║  DO NOT CHANGE FUNCTION SIGNATURES.                             ║
║  DO NOT REMOVE ANY FUNCTION.                                    ║
║  DO NOT RENAME CLS_CLASS_MAPPING or SEG_CLASS_MAPPING.          ║
║  You may add helper functions / imports as needed.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════
# CLASS MAPPINGS
# ═══════════════════════════════════════════════════════════════════

# Classification model output indices → canonical class names
# ResNet-50 was trained with: {1:0, 8:1, 7:2, 2:3, 9:4}
CLS_CLASS_MAPPING: Dict[int, str] = {
    0: "short sleeve top",
    1: "trousers",
    2: "shorts",
    3: "long sleeve top",
    4: "skirt",
}

# YOLOv8 output indices → canonical class names (no background)
SEG_CLASS_MAPPING: Dict[int, str] = {
    0: "short sleeve top",
    1: "trousers",
    2: "shorts",
    3: "long sleeve top",
    4: "skirt",
}


# ═══════════════════════════════════════════════════════════════════
# TUNED THRESHOLDS (update after threshold tuning)
# ═══════════════════════════════════════════════════════════════════
# Order matches CLS_CLASS_MAPPING:
# [short_sleeve_top, trousers, shorts, long_sleeve_top, skirt]
CLS_THRESHOLDS = [0.5, 0.5, 0.5, 0.5, 0.5]


# ═══════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _find_weights(folder: Path, stem: str) -> Path:
    """Return the first existing weights file matching stem.pt or stem.pth."""
    for ext in (".pt", ".pth"):
        candidate = folder / "model_files" / (stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights file found for '{stem}' in {folder / 'model_files'}"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# TASK 3.1 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

# Transform for classification inference
_CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_classification_model(folder: str, device: str) -> Any:
    """
    Load ResNet-50 classification model.

    Parameters
    ----------
    folder : str
        Absolute path to submission folder containing model_files/cls.pt
    device : str
        PyTorch device string e.g. "cuda" or "cpu"

    Returns
    -------
    dict with keys 'model' and 'device'
    """
    folder = Path(folder)
    weights_path = _find_weights(folder, "cls")

    # Build ResNet-50 with 5-class output head
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"Classification model loaded from {weights_path}")
    return {"model": model, "device": device}


def predict_classification(
    model: Any,
    images: List[Image.Image]
) -> List[Dict]:
    """
    Run multi-label classification on a list of PIL images.

    Parameters
    ----------
    model : dict
        Object returned by load_classification_model()
    images : list of PIL.Image.Image
        RGB PIL images

    Returns
    -------
    list of dict, each with key "labels":
        [{"labels": [0, 1, 0, 0, 1]}, ...]
        5 binary ints per image in CLS_CLASS_MAPPING order
    """
    net    = model["model"]
    device = model["device"]

    results = []

    # Process in batches of 32
    batch_size = 32
    for batch_start in range(0, len(images), batch_size):
        batch_imgs = images[batch_start: batch_start + batch_size]

        # Convert PIL images to tensor batch
        tensors = torch.stack([
            _CLS_TRANSFORM(img.convert("RGB"))
            for img in batch_imgs
        ]).to(device)

        with torch.no_grad():
            logits = net(tensors)
            probs  = torch.sigmoid(logits).cpu().numpy()

        for prob_row in probs:
            # Apply per-class thresholds
            binary_labels = [
                1 if prob_row[i] >= CLS_THRESHOLDS[i] else 0
                for i in range(5)
            ]
            results.append({"labels": binary_labels})

    return results


# ═══════════════════════════════════════════════════════════════════
# TASK 3.2 — DETECTION + INSTANCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def load_detection_model(folder: str, device: str) -> Any:
    """
    Load YOLOv8s-seg detection + segmentation model.

    Parameters
    ----------
    folder : str
        Absolute path to submission folder containing model_files/seg.pt
    device : str
        PyTorch device string e.g. "cuda" or "cpu"

    Returns
    -------
    dict with keys 'model' and 'device'
    """
    folder = Path(folder)
    weights_path = _find_weights(folder, "seg")

    model = YOLO(str(weights_path))

    print(f"Detection model loaded from {weights_path}")
    return {"model": model, "device": device}


def predict_detection_segmentation(
    model: Any,
    images: List[Image.Image],
) -> List[Dict]:
    """
    Run detection + instance segmentation on a list of PIL images.

    Parameters
    ----------
    model : dict
        Object returned by load_detection_model()
    images : list of PIL.Image.Image
        RGB PIL images

    Returns
    -------
    list of dict, each with keys:
        "boxes"  : [[x1, y1, x2, y2], ...]  float coords
        "scores" : [float, ...]              confidence in [0, 1]
        "labels" : [int, ...]                class indices per SEG_CLASS_MAPPING
        "masks"  : [np.ndarray(H, W), ...]   binary uint8, ORIGINAL image size
    """
    yolo   = model["model"]
    device = model["device"]

    # Map device string to YOLO device format
    yolo_device = 0 if device == "cuda" else "cpu"

    results = []

    for img in images:
        orig_w, orig_h = img.size  # original image dimensions

        # Run YOLO inference
        preds = yolo(
            img,
            imgsz   = 640,
            conf    = 0.25,
            device  = yolo_device,
            verbose = False
        )

        boxes_out  = []
        scores_out = []
        labels_out = []
        masks_out  = []

        for pred in preds:
            det_boxes  = pred.boxes
            det_masks  = pred.masks

            if det_boxes is None or len(det_boxes) == 0:
                continue

            for i in range(len(det_boxes)):
                box   = det_boxes[i]
                cls   = int(box.cls.item())
                score = float(box.conf.item())
                xyxy  = box.xyxy[0].tolist()

                x1, y1, x2, y2 = xyxy

                # Clamp coords to image bounds
                x1 = max(0.0, min(x1, float(orig_w)))
                y1 = max(0.0, min(y1, float(orig_h)))
                x2 = max(0.0, min(x2, float(orig_w)))
                y2 = max(0.0, min(y2, float(orig_h)))

                boxes_out.append([x1, y1, x2, y2])
                scores_out.append(score)
                labels_out.append(cls)

                # ── Mask at ORIGINAL resolution ──────────────────
                if det_masks is not None and i < len(det_masks):
                    # YOLO returns masks at 640x640 — resize to original
                    mask_data = det_masks[i].data  # tensor [1, H, W]
                    mask_np   = mask_data.squeeze().cpu().numpy()

                    # Resize mask back to original image size
                    mask_pil  = Image.fromarray(
                        (mask_np * 255).astype(np.uint8)
                    ).resize(
                        (orig_w, orig_h),
                        Image.NEAREST
                    )
                    mask_bin  = (np.array(mask_pil) > 127).astype(np.uint8)
                else:
                    # No mask — return empty binary mask
                    mask_bin = np.zeros(
                        (orig_h, orig_w), dtype=np.uint8
                    )

                masks_out.append(mask_bin)

        results.append({
            "boxes" : boxes_out,
            "scores": scores_out,
            "labels": labels_out,
            "masks" : masks_out,
        })

    return results

