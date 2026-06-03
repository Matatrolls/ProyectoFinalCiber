"""
Preprocesa imagenes de slots de digitos y extrae caracteristicas estructurales
necesarias para la deteccion de anomalias (componentes conectados, tamano, etc.).
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field


IMG_SIZE = 32


@dataclass
class SlotFeatures:
    tensor:           torch.Tensor        # [1, 32, 32] imagen normalizada (entrada al modelo)
    is_blank:         bool                # campo practicamente vacio
    num_components:   int                 # n componentes conectados significativos
    fill_ratio:       float               # fraccion de pixeles de tinta
    bbox_w_ratio:     float               # ancho del BBox principal / ancho slot
    bbox_h_ratio:     float               # alto del BBox principal / alto slot
    raw_crop:         np.ndarray = field(default=None, repr=False)
    binary:           np.ndarray = field(default=None, repr=False)


def preprocess_slot(crop: np.ndarray,
                    min_area: int = 12,
                    max_area: int = 900) -> SlotFeatures:
    """
    Preprocesa un recorte de slot de digito y extrae caracteristicas.
    """
    if crop is None or crop.size == 0:
        blank_t = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        return SlotFeatures(blank_t, True, 0, 0.0, 0.0, 0.0)

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.uint8)

    mean_val = float(gray.mean())
    if mean_val < 128:
        gray = 255 - gray

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    total_pixels = binary.size
    fill_ratio   = float(binary.sum()) / (255 * total_pixels)

    is_blank = fill_ratio < 0.03

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    significant = []
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            significant.append(stats[lbl])

    num_components = len(significant)

    if significant:
        main = max(significant, key=lambda s: s[cv2.CC_STAT_AREA])
        bbox_w_ratio = main[cv2.CC_STAT_WIDTH]  / IMG_SIZE
        bbox_h_ratio = main[cv2.CC_STAT_HEIGHT] / IMG_SIZE
    else:
        bbox_w_ratio = 0.0
        bbox_h_ratio = 0.0

    norm = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(norm).unsqueeze(0)    # [1, H, W]

    return SlotFeatures(
        tensor        = tensor,
        is_blank      = is_blank,
        num_components= num_components,
        fill_ratio    = fill_ratio,
        bbox_w_ratio  = bbox_w_ratio,
        bbox_h_ratio  = bbox_h_ratio,
        raw_crop      = crop,
        binary        = binary,
    )


def batch_preprocess(crops: list[np.ndarray],
                     min_area: int = 12,
                     max_area: int = 900) -> list[SlotFeatures]:
    return [preprocess_slot(c, min_area, max_area) for c in crops]