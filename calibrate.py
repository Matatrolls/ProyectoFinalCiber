"""
Herramienta visual para verificar y ajustar las coordenadas de pages_config.json.

Uso:
    python calibrate.py --pdf template.pdf --page 2
    python calibrate.py --pdf template.pdf --page 1 --out calibration_checks/
"""

import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path


COLORS = {
    "nivelacion":        (0,   200,  0),
    "preferente_full":   (200, 100,  0),
    "sin_preferente":    (0,   100, 200),
    "totales_nacionales":(150,   0, 200),
}
DEFAULT_COLOR = (0, 0, 200)
TEMPLATE_W, TEMPLATE_H = 532, 1568


def scale_bbox(bbox, img_w, img_h):
    sx, sy = img_w / TEMPLATE_W, img_h / TEMPLATE_H
    x1, y1, x2, y2 = bbox
    return int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)


def draw_page(page_img, page_num, config, zoom=2.0):
    h, w = page_img.shape[:2]
    canvas = page_img.copy()

    page_index = {p["page"]: p for p in config["pages"]}
    if page_num not in page_index:
        print(f"Pagina {page_num} no encontrada en config.")
        return canvas

    page_cfg  = page_index[page_num]
    tpl_name  = page_cfg["template"]
    template  = config["templates"][tpl_name]
    color     = COLORS.get(tpl_name, DEFAULT_COLOR)

    def draw_field(bbox, label, col=None):
        x1, y1, x2, y2 = scale_bbox(bbox, w, h)
        c = col or color
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 1)
        cv2.putText(canvas, label[:18], (x1+1, y1+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, c, 1, cv2.LINE_AA)

    if tpl_name == "skip":
        cv2.putText(canvas, "SKIP - sin campos", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)

    elif tpl_name in ("nivelacion", "sin_preferente", "totales_nacionales", "totales_indigenas"):
        for fd in template["fields"]:
            draw_field(fd["bbox"], fd["name"])

    else:
        r0 = template["row0"]
        draw_field(r0["bbox"], r0["name"], col=(0, 180, 180))

        tr = template["total_row"]
        draw_field(tr["bbox"], tr["name"], col=(0,0,180))
        

    info = (f"Pg {page_num} | {page_cfg.get('section','')[:26]} | "
            f"template={tpl_name} | {page_cfg.get('list_name','')[:20]}")
    cv2.rectangle(canvas, (0, 0), (w, 16), (30,30,30), -1)
    cv2.putText(canvas, info, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (230,230,230), 1, cv2.LINE_AA)

    if zoom != 1.0:
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return canvas


def main():
    p = argparse.ArgumentParser(description="Calibracion visual coordenadas E-14")
    p.add_argument("--pdf",    required=True,            help="PDF o ZIP plantilla")
    p.add_argument("--config", default="config/pages_config.json")
    p.add_argument("--page",   type=int, default=None,   help="Pagina a revisar (1-30). Sin arg: todas.")
    p.add_argument("--out",    default="output/calibration", help="Dir para guardar imagenes")
    p.add_argument("--zoom",   type=float, default=2.0)
    p.add_argument("--show",   action="store_true",      help="Mostrar ventana interactiva")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    from src.pdf_processor import load_pdf
    pages = load_pdf(args.pdf)
    print(f"Paginas cargadas: {len(pages)}")

    os.makedirs(args.out, exist_ok=True)

    page_nums = [args.page] if args.page else list(range(1, len(pages) + 1))

    for pg_num in page_nums:
        if pg_num > len(pages):
            continue
        page_img  = pages[pg_num - 1]
        annotated = draw_page(page_img, pg_num, config, zoom=args.zoom)

        out_path = os.path.join(args.out, f"page_{pg_num:02d}.png")
        cv2.imwrite(out_path, annotated)
        print(f"  Guardado: {out_path}")

        if args.show:
            cv2.imshow(f"Pagina {pg_num}", annotated)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    print(f"\nImagenes de calibracion en: {args.out}/")
    print("Ajusta los valores bbox en config/pages_config.json si es necesario.")


if __name__ == "__main__":
    main()
