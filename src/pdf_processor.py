"""
Extrae paginas de PDFs o del formato ZIP-de-JPEGs usado en el formulario E-14.
Devuelve lista de imagenes BGR (numpy) en orden de pagina.
"""

import os
import io
import json
import zipfile
import tempfile
import numpy as np
import cv2
from pathlib import Path


TEMPLATE_W = 532
TEMPLATE_H = 1568


def _load_jpeg_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _resize_to_template(img: np.ndarray) -> np.ndarray:
    if img.shape[1] == TEMPLATE_W and img.shape[0] == TEMPLATE_H:
        return img
    return cv2.resize(img, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)


def load_zip_pdf(path: str) -> list[np.ndarray]:
    """
    Carga el formato ZIP-de-JPEGs del E-14 (paginas 1.jpeg ... 30.jpeg).
    """
    pages: list[tuple[int, np.ndarray]] = []

    with zipfile.ZipFile(path, "r") as zf:
        manifest_raw = None
        try:
            manifest_raw = json.loads(zf.read("manifest.json"))
        except KeyError:
            pass

        if manifest_raw:
            for page_info in manifest_raw["pages"]:
                page_num  = page_info["page_number"]
                img_path  = page_info["image"]["path"]
                img_bytes = zf.read(img_path)
                img       = _load_jpeg_from_bytes(img_bytes)
                if img is not None:
                    pages.append((page_num, _resize_to_template(img)))
        else:
            for name in sorted(zf.namelist()):
                stem = Path(name).stem
                if stem.isdigit() and name.endswith((".jpeg", ".jpg", ".png")):
                    img_bytes = zf.read(name)
                    img = _load_jpeg_from_bytes(img_bytes)
                    if img is not None:
                        pages.append((int(stem), _resize_to_template(img)))

    pages.sort(key=lambda t: t[0])
    return [img for _, img in pages]


def load_real_pdf(path: str, dpi: int = 200) -> list[np.ndarray]:
    """
    Convierte PDF real a imagenes usando pdf2image (requiere Poppler).
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("Instala pdf2image y Poppler para procesar PDFs reales.")

    pil_pages = convert_from_path(path, dpi=dpi)
    pages = []
    for pil_img in pil_pages:
        arr = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        pages.append(_resize_to_template(bgr))
    return pages


def load_pdf(path: str, dpi: int = 200) -> list[np.ndarray]:
    """
    Detecta formato automaticamente (ZIP o PDF real) y carga las paginas.
    Siempre devuelve imagenes redimensionadas a TEMPLATE_W x TEMPLATE_H.
    """
    magic = open(path, "rb").read(4)
    if magic[:2] == b"PK":
        return load_zip_pdf(path)
    else:
        return load_real_pdf(path, dpi=dpi)


def load_pdf_batch(pdf_paths: list[str], dpi: int = 200) -> dict[str, list[np.ndarray]]:
    """
    Carga multiples PDFs. Devuelve {nombre_archivo: [paginas]}.
    """
    result = {}
    for p in pdf_paths:
        name = Path(p).name
        try:
            pages = load_pdf(p, dpi=dpi)
            result[name] = pages
            print(f"  [OK] {name:40s} — {len(pages)} paginas")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
    return result