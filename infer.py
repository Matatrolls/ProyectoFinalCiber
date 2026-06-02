"""
Pipeline de inferencia para deteccion de anomalias en formularios E-14.

Uso:
    python infer.py --pdfs pdfs/           # directorio con PDFs
    python infer.py --pdfs E14_001.pdf     # un solo archivo
    python infer.py --pdfs pdfs/ --model models/digit_cnn_best.pt --threshold 0.90
"""

import argparse
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia anomalias E-14")
    p.add_argument("--pdfs",      required=True,                   help="Archivo o directorio de PDFs")
    p.add_argument("--model",     default="models/digit_cnn_best.pt")
    p.add_argument("--config",    default="config/pages_config.json")
    p.add_argument("--out",       default="output/")
    p.add_argument("--threshold", type=float, default=0.85,        help="Confianza minima (0-1)")
    p.add_argument("--entropy",   type=float, default=1.20,        help="Entropia maxima")
    p.add_argument("--margin",    type=float, default=0.25,        help="Margen minimo top1-top2")
    p.add_argument("--device",    default="auto")
    p.add_argument("--batch",     type=int,   default=512,         help="Slots por lote GPU")
    p.add_argument("--only_anomalies", action="store_true",        help="Reportar solo anomalias")
    return p.parse_args()


def resolve_pdfs(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        pdfs = sorted(glob.glob(str(p / "*.pdf"))) + sorted(glob.glob(str(p / "*.PDF")))
        return pdfs
    raise FileNotFoundError(f"No encontrado: {path}")


def process_pdf(pdf_path: str, extractor, segmenter_fn, detector,
                crops_dir: str, batch_size: int, only_anomalies: bool):
    from src.pdf_processor    import load_pdf
    from src.roi_extractor    import ROIExtractor
    from src.digit_segmenter  import preprocess_slot
    from src.report_generator import build_records, AnomalyRecord

    pdf_name = Path(pdf_path).name

    pages = load_pdf(pdf_path)
    if not pages:
        print(f"  [WARN] Sin paginas: {pdf_name}")
        return []

    rois = extractor.extract_document(pages)
    if not rois:
        return []

    slot_features = [preprocess_slot(roi.crop) for roi in rois]

    results = []
    for i in range(0, len(slot_features), batch_size):
        batch = slot_features[i:i+batch_size]
        results.extend(detector.classify_batch(batch))

    records = build_records(pdf_name, rois, results, crops_dir)

    if only_anomalies:
        records = [r for r in records if r.status == "ANOMALIA"]

    return records


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    pdf_list = resolve_pdfs(args.pdfs)
    if not pdf_list:
        print("Sin PDFs encontrados.")
        return
    print(f"PDFs a procesar: {len(pdf_list)}")

    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        print(f"[ERROR] pages_config.json mal formado. Debe ser un objeto JSON, no lista.")
        return
    thresholds = cfg.get("thresholds", {})

    from src.model           import load_model
    from src.roi_extractor   import ROIExtractor
    from src.anomaly_detector import AnomalyDetector
    from src.report_generator import export_all

    if not os.path.exists(args.model):
        print(f"[ERROR] Modelo no encontrado: {args.model}")
        print("  Ejecuta primero: python train.py --dataset dataset/")
        return

    model     = load_model(args.model, device=device)
    extractor = ROIExtractor(args.config)
    detector  = AnomalyDetector(
        model          = model,
        device         = device,
        confidence_min = args.threshold,
        entropy_max    = args.entropy,
        margin_min     = args.margin,
        max_components = thresholds.get("max_components_per_slot", 1),
    )

    crops_dir   = os.path.join(args.out, "crops")
    reports_dir = os.path.join(args.out, "reports")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_records = []

    for pdf_path in tqdm(pdf_list, desc="Procesando PDFs"):
        records = process_pdf(
            pdf_path      = pdf_path,
            extractor     = extractor,
            segmenter_fn  = None,
            detector      = detector,
            crops_dir     = crops_dir,
            batch_size    = args.batch,
            only_anomalies= args.only_anomalies,
        )
        all_records.extend(records)
        anomalias = sum(1 for r in records if r.status == "ANOMALIA")
        print(f"  {Path(pdf_path).name:40s} | registros={len(records):5d} | anomalias={anomalias:4d}")

    if all_records:
        csv_p, xlsx_p = export_all(all_records, reports_dir)
        print(f"\nReportes generados:")
        print(f"  CSV  → {csv_p}")
        print(f"  XLSX → {xlsx_p}")
    else:
        print("\nSin registros para reportar.")

    total_anomalias = sum(1 for r in all_records if r.status == "ANOMALIA")
    print(f"\nResumen: {len(pdf_list)} PDFs | {len(all_records)} registros | {total_anomalias} anomalias")


if __name__ == "__main__":
    main()