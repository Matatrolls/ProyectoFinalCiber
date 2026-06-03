"""
Detector de anomalias multi-nivel para slots de digitos en formularios E-14.

Niveles de deteccion:
  1. Anomalia Isolation Forest (score de decision < umbral)
  2. Entropia alta             (distribucion demasiado uniforme)
  3. Margen insuficiente       (diferencia top1 - top2 < umbral)
  4. Componentes multiples     (mas de 1 componente conectado en el slot)
  5. Tamano anormal            (digito demasiado pequeno o grande)
  6. Campo fragmentado         (componente con bbox muy alargado)
"""

from __future__ import annotations
import math
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .digit_segmenter import SlotFeatures
from .model import IsolationForestAnomalyDetector


CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "dot"]


@dataclass
class SlotResult:
    predicted_class:  int
    predicted_label:  str
    confidence:       float
    entropy:          float
    margin:           float
    softmax:          list[float]
    top3:             list[tuple[str, float]]

    is_anomaly:       bool
    anomaly_reasons:  list[str]

    is_blank:         bool
    num_components:   int
    fill_ratio:       float


def _entropy(probs: torch.Tensor) -> float:
    eps = 1e-9
    p   = probs.clamp(eps, 1.0)
    return float(-(p * p.log()).sum().item())


def _top3(probs: torch.Tensor) -> list[tuple[str, float]]:
    vals, idxs = probs.topk(3)
    return [(CLASSES[i.item()], float(v.item())) for i, v in zip(idxs, vals)]


class AnomalyDetector:

    def __init__(self,
                 model: IsolationForestAnomalyDetector,
                 device: str = "cpu",
                 confidence_min: float   = 0.85,
                 entropy_max: float      = 1.20,
                 margin_min: float       = 0.25,
                 max_components: int     = 1,
                 min_bbox_ratio: float   = 0.08,
                 max_bbox_ratio: float   = 0.95):

        self.model         = model
        self.device        = device
        self.conf_min      = confidence_min
        self.entropy_max   = entropy_max
        self.margin_min    = margin_min
        self.max_comp      = max_components
        self.min_bbox      = min_bbox_ratio
        self.max_bbox      = max_bbox_ratio

    def classify_slot(self, sf: SlotFeatures) -> SlotResult:
        """Clasifica un slot y evalua todas las condiciones de anomalia."""

        if sf.is_blank:
            dot_idx = CLASSES.index("dot")
            soft    = [0.0] * len(CLASSES)
            soft[dot_idx] = 1.0
            return SlotResult(
                predicted_class  = dot_idx,
                predicted_label  = "dot",
                confidence       = 1.0,
                entropy          = 0.0,
                margin           = 1.0,
                softmax          = soft,
                top3             = [("dot", 1.0), (CLASSES[0], 0.0), (CLASSES[1], 0.0)],
                is_anomaly       = False,
                anomaly_reasons  = [],
                is_blank         = True,
                num_components   = sf.num_components,
                fill_ratio       = sf.fill_ratio,
            )

        x_flat = sf.tensor.cpu().numpy().flatten().reshape(1, -1)
        iforest_pred, score, probs_batch, preds_batch = self.model.predict(x_flat)

        probs = torch.tensor(probs_batch[0])
        conf    = float(probs.max().item())
        pred    = int(probs.argmax().item())
        ent     = _entropy(probs)
        sorted_p, _ = probs.sort(descending=True)
        margin  = float((sorted_p[0] - sorted_p[1]).item())

        reasons: list[str] = []

        if score[0] < self.conf_min:
            reasons.append(f"isolation_forest_anomaly ({score[0]:.4f})")
        if sf.num_components > self.max_comp:
            reasons.append(f"multiples_componentes ({sf.num_components})")
        if sf.num_components > 0:
            if sf.bbox_w_ratio > self.max_bbox or sf.bbox_h_ratio > self.max_bbox:
                reasons.append("caracter_fusionado")
            if sf.bbox_w_ratio < self.min_bbox and sf.bbox_h_ratio < self.min_bbox:
                reasons.append("fragmento_pequeno")

        return SlotResult(
            predicted_class  = pred,
            predicted_label  = CLASSES[pred],
            confidence       = conf,
            entropy          = ent,
            margin           = margin,
            softmax          = probs.tolist(),
            top3             = _top3(probs),
            is_anomaly       = bool(reasons),
            anomaly_reasons  = reasons,
            is_blank         = sf.is_blank,
            num_components   = sf.num_components,
            fill_ratio       = sf.fill_ratio,
        )

    def classify_batch(self, slot_features: list[SlotFeatures]) -> list[SlotResult]:
        """Clasifica una lista de slots eficientemente."""
        non_blank_idx = [i for i, sf in enumerate(slot_features) if not sf.is_blank]
        results: list[Optional[SlotResult]] = [None] * len(slot_features)

        for i, sf in enumerate(slot_features):
            if sf.is_blank:
                results[i] = self.classify_slot(sf)

        if non_blank_idx:
            x_batch = np.stack([slot_features[i].tensor.cpu().numpy().flatten() for i in non_blank_idx])
            
            iforest_preds, scores, probs_batch, preds_batch = self.model.predict(x_batch)

            for batch_pos, orig_i in enumerate(non_blank_idx):
                sf    = slot_features[orig_i]
                probs = torch.tensor(probs_batch[batch_pos])

                conf   = float(probs.max().item())
                pred   = int(probs.argmax().item())
                ent    = _entropy(probs)
                sp, _  = probs.sort(descending=True)
                margin = float((sp[0] - sp[1]).item())
                score  = scores[batch_pos]

                reasons: list[str] = []
                if score < self.conf_min:
                    reasons.append(f"isolation_forest_anomaly ({score:.4f})")
                if sf.num_components > self.max_comp:
                    reasons.append(f"multiples_componentes ({sf.num_components})")
                if sf.num_components > 0:
                    if sf.bbox_w_ratio > self.max_bbox or sf.bbox_h_ratio > self.max_bbox:
                        reasons.append("caracter_fusionado")
                    if sf.bbox_w_ratio < self.min_bbox and sf.bbox_h_ratio < self.min_bbox:
                        reasons.append("fragmento_pequeno")

                results[orig_i] = SlotResult(
                    predicted_class = pred,
                    predicted_label = CLASSES[pred],
                    confidence      = conf,
                    entropy         = ent,
                    margin          = margin,
                    softmax         = probs.tolist(),
                    top3            = _top3(probs),
                    is_anomaly      = bool(reasons),
                    anomaly_reasons = reasons,
                    is_blank        = sf.is_blank,
                    num_components  = sf.num_components,
                    fill_ratio      = sf.fill_ratio,
                )

        return results