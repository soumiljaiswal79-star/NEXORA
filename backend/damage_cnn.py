"""
Real MobileNetV2 damage classifier for field-uploaded photos.

Loads torchvision MobileNetV2 (ImageNet weights) once at startup and, for every
uploaded photo, produces:
  - a damage severity label (None / Low / Medium / High)
  - a numeric damage_score (0-100) that gets fed straight into the XGBoost
    calibration model as its 7th feature
  - the top-3 ImageNet predictions so the operator can see what the CNN saw

The score combines three genuine signals:
  1. Summed softmax probability over ~40 ImageNet classes that visually match
     road-damage / flood / landslide field photos (mud, water, rubble, cliffs).
  2. Sobel edge density  — proxy for rubble, cracks, potholes.
  3. Dark-region ratio   — proxy for shadows, water, washed-out surface.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

_STATE: dict = {"model": None, "categories": [], "damage_idx": None, "preprocess": None}

# ImageNet class names whose visual content overlaps with road-damage,
# flooding, landslides and generic infrastructure failure imagery.
_DAMAGE_KEYWORDS = (
    "mud", "lake", "seashore", "geyser", "cliff", "valley", "alp",
    "volcano", "ravine", "canyon", "breakwater", "dam", "brick",
    "stone wall", "worm fence", "picket fence", "wreck", "barn",
    "boathouse", "shipwreck", "dock", "quicksand",
)


def _lazy_init() -> None:
    if _STATE["model"] is not None:
        return
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights)
    model.eval()
    categories = list(weights.meta["categories"])
    damage_idx = [i for i, name in enumerate(categories)
                  if any(kw in name.lower() for kw in _DAMAGE_KEYWORDS)]
    _STATE["model"] = model
    _STATE["categories"] = categories
    _STATE["damage_idx"] = damage_idx
    _STATE["preprocess"] = weights.transforms()


def _image_stats(img: Image.Image) -> tuple[float, float]:
    """Return (edge_density, dark_ratio) — both in [0, 1]."""
    grey = np.asarray(img.convert("L").resize((160, 160)), dtype=np.float32) / 255.0
    # Sobel-ish: absolute gradient magnitude.
    dx = np.abs(np.diff(grey, axis=1))
    dy = np.abs(np.diff(grey, axis=0))
    edge = float(np.clip((dx.mean() + dy.mean()) / 0.30, 0.0, 1.0))
    dark = float((grey < 0.30).mean())
    return edge, dark


def _decode(image_data: str) -> Optional[Image.Image]:
    if not image_data:
        return None
    payload = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
    try:
        raw = base64.b64decode(payload, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _severity_from_score(score: float) -> str:
    if score >= 72:
        return "High"
    if score >= 48:
        return "Medium"
    if score >= 24:
        return "Low"
    return "None"


def classify(image_data: Optional[str]) -> dict:
    """Run MobileNetV2 + image stats on a base64-encoded image."""
    _lazy_init()
    img = _decode(image_data) if image_data else None
    if img is None:
        return {"severity": None, "score": 0.0, "top_classes": [], "used": False,
                "edge_density": 0.0, "dark_ratio": 0.0}
    tensor = _STATE["preprocess"](img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_STATE["model"](tensor)[0], dim=0)
    damage_prob = float(probs[_STATE["damage_idx"]].sum())
    top_p, top_i = torch.topk(probs, 3)
    top_classes = [{"label": _STATE["categories"][int(i)], "confidence": round(float(p), 3)}
                   for i, p in zip(top_i, top_p)]
    edge, dark = _image_stats(img)
    # Blend the three real signals into a 0-100 damage score.
    score = 100.0 * (0.55 * damage_prob + 0.30 * edge + 0.15 * dark)
    score = float(np.clip(score, 0.0, 100.0))
    return {
        "severity": _severity_from_score(score),
        "score": round(score, 1),
        "top_classes": top_classes,
        "used": True,
        "edge_density": round(edge, 3),
        "dark_ratio": round(dark, 3),
        "damage_prob": round(damage_prob, 3),
    }


def warmup() -> None:
    """Force weight load + a single forward pass so first request is fast."""
    _lazy_init()
    with torch.no_grad():
        _STATE["model"](torch.zeros(1, 3, 224, 224))
