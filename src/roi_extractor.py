"""
Extrae regiones de interes (ROI) de cada pagina del formulario E-14
usando las coordenadas definidas en pages_config.json.

Cada ROI representa un campo numerico (potencialmente 1-3 digitos).
"""

import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field


TEMPLATE_W = 532
TEMPLATE_H = 1568


@dataclass
class FieldROI:
    page:        int
    section:     str
    list_name:   str
    field_name:  str
    candidate:   int | None        # numero de candidato si aplica
    col_id:      str | None        # left / center / right
    digit_slot:  int               # 0=hundreds, 1=tens, 2=units dentro del campo
    bbox:        tuple[int,int,int,int]    # (x1,y1,x2,y2) en imagen real
    crop:        np.ndarray = field(default=None, repr=False)


def _scale_bbox(bbox: list[int], img_w: int, img_h: int):
    sx = img_w / TEMPLATE_W
    sy = img_h / TEMPLATE_H
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1 * sx)),
        max(0, int(y1 * sy)),
        min(img_w, int(x2 * sx)),
        min(img_h, int(y2 * sy)),
    )


def _split_into_slots(img: np.ndarray, num_digits: int = 3) -> list[np.ndarray]:
    """Divide un campo horizontal en num_digits sub-regiones iguales."""
    h, w = img.shape[:2]
    sw   = w // num_digits
    slots = []
    for i in range(num_digits):
        x1 = i * sw
        x2 = (i + 1) * sw if i < num_digits - 1 else w
        slots.append(img[:, x1:x2])
    return slots


def _crop(img: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()


class ROIExtractor:

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.templates   = cfg["templates"]
        self.page_index  = {p["page"]: p for p in cfg["pages"]}
        self.num_classes = len(cfg["classes"])

    def extract_page(self, page_img: np.ndarray, page_number: int) -> list[FieldROI]:
        """
        Extrae todos los FieldROI de una pagina.
        Cada FieldROI contiene la imagen recortada del slot individual (1 digito).
        """
        if page_number not in self.page_index:
            return []

        page_cfg  = self.page_index[page_number]
        template  = self.templates[page_cfg["template"]]
        section   = page_cfg.get("section", "")
        list_name = page_cfg.get("list_name", "")

        if page_cfg["template"] == "skip":
            return []

        h, w = page_img.shape[:2]
        rois: list[FieldROI] = []

        if page_cfg["template"] in ("nivelacion", "sin_preferente", "totales_nacionales"):
            for field_def in template["fields"]:
                bbox    = _scale_bbox(field_def["bbox"], w, h)
                field_img = _crop(page_img, bbox)
                nd      = field_def.get("num_digits", 3)
                slots   = _split_into_slots(field_img, nd)
                for slot_i, slot_img in enumerate(slots):
                    rois.append(FieldROI(
                        page       = page_number,
                        section    = section,
                        list_name  = list_name,
                        field_name = field_def["name"],
                        candidate  = None,
                        col_id     = None,
                        digit_slot = slot_i,
                        bbox       = bbox,
                        crop       = slot_img,
                    ))

        else:
            row0    = template["row0"]
            bbox_r0 = _scale_bbox(row0["bbox"], w, h)
            img_r0  = _crop(page_img, bbox_r0)
            for slot_i, slot_img in enumerate(_split_into_slots(img_r0, 3)):
                rois.append(FieldROI(
                    page       = page_number,
                    section    = section,
                    list_name  = list_name,
                    field_name = row0["name"],
                    candidate  = 0,
                    col_id     = None,
                    digit_slot = slot_i,
                    bbox       = bbox_r0,
                    crop       = slot_img,
                ))

        

            total_def  = template["total_row"]
            bbox_tot   = _scale_bbox(total_def["bbox"], w, h)
            img_tot    = _crop(page_img, bbox_tot)
            for slot_i, slot_img in enumerate(_split_into_slots(img_tot, 3)):
                rois.append(FieldROI(
                    page       = page_number,
                    section    = section,
                    list_name  = list_name,
                    field_name = total_def["name"],
                    candidate  = None,
                    col_id     = None,
                    digit_slot = slot_i,
                    bbox       = bbox_tot,
                    crop       = slot_img,
                ))

        return rois

    def extract_document(self, pages: list[np.ndarray]) -> list[FieldROI]:
        all_rois = []
        for pg_idx, page_img in enumerate(pages):
            pg_num = pg_idx + 1
            all_rois.extend(self.extract_page(page_img, pg_num))
        return all_rois