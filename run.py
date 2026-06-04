"""
Orquestador principal E-14 — ejecuta TODO en un solo comando.

Uso minimo:
    python run.py --dataset dataset/ --pdfs pdfs/

Con opciones:
    python run.py --dataset dataset/ --pdfs pdfs/ --template plantilla.pdf \\
                  --epochs 40 --threshold 0.85 --device auto --only_anomalies

Fases automaticas:
    [1] Instala dependencias de requirements.txt (si faltan)
    [2] Entrena el Isolation Forest (omitido si ya existe models/isolation_forest.joblib y no se pasa --retrain)
    [3] Calibracion     (opcional, solo si se pasa --template)
    [4] Inferencia sobre los PDFs
    [5] Muestra ruta de los reportes generados
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def banner(msg: str):
    w = 60
    print()
    print("=" * w)
    print(f"  {msg}")
    print("=" * w)


def run(cmd: list[str], check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Comando fallido (codigo {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def install_deps(req_file: str = "requirements.txt"):
    banner("FASE 1 — Instalando dependencias")
    if not Path(req_file).exists():
        print(f"  [WARN] {req_file} no encontrado, saltando instalacion.")
        return
    run([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])
    print("  Dependencias OK.")


def check_dataset(dataset_dir: str):
    p = Path(dataset_dir)
    if not p.exists():
        print(f"\n[ERROR] Dataset no encontrado: {dataset_dir}")
        print("  Crea la carpeta con subcarpetas 0/ 1/ ... 9/ dot/")
        sys.exit(1)
    classes = [d.name for d in p.iterdir() if d.is_dir()]
    expected = {str(i) for i in range(10)} | {"dot", "lines"}
    missing  = expected - set(classes)
    if missing:
        print(f"\n[WARN] Clases faltantes en dataset: {sorted(missing)}")
    total = sum(len(list(p.glob("**/*.*"))) for p in p.iterdir() if p.is_dir())
    print(f"  Dataset: {len(classes)} clases | ~{total} imagenes")


def do_train(args):
    banner("FASE 2 — Entrenamiento de Isolation Forest")
    model_path = Path(args.model)
    if model_path.exists() and not args.retrain:
        print(f"  Modelo ya existe: {model_path}")
        print("  Saltando entrenamiento.  (usa --retrain para forzar)")
        return

    check_dataset(args.dataset)
    cmd = [
        sys.executable, "train.py",
        "--dataset", args.dataset,
        "--epochs",  str(args.epochs),
        "--batch",   str(args.batch),
        "--lr",      str(args.lr),
        "--val",     str(args.val),
        "--workers", str(args.workers),
        "--device",  args.device,
        "--out",     str(model_path.parent),
        "--contamination", str(args.contamination),
    ]
    run(cmd)


def do_calibrate(args):
    if not args.template:
        return
    banner("FASE 3 — Calibracion de coordenadas")
    if not Path(args.template).exists():
        print(f"  [WARN] Plantilla no encontrada: {args.template}. Saltando calibracion.")
        return
    cmd = [
        sys.executable, "calibrate.py",
        "--pdf",    args.template,
        "--config", args.config,
        "--out",    "output/calibration",
        "--zoom",   str(args.zoom),
    ]
    run(cmd)
    print(f"  Imagenes de calibracion en: output/calibration/")


def do_infer(args):
    banner("FASE 4 — Inferencia y deteccion de anomalias")
    if not Path(args.pdfs).exists():
        print(f"\n[ERROR] Ruta de PDFs no encontrada: {args.pdfs}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"\n[ERROR] Modelo no encontrado: {args.model}")
        print("  Ejecuta con --dataset para entrenar primero.")
        sys.exit(1)

    cmd = [
        sys.executable, "infer.py",
        "--pdfs",      args.pdfs,
        "--model",     args.model,
        "--config",    args.config,
        "--out",       args.out,
        "--threshold", str(args.threshold),
        "--entropy",   str(args.entropy),
        "--margin",    str(args.margin),
        "--device",    args.device,
        "--batch",     str(args.batch_infer),
    ]
    if args.only_anomalies:
        cmd.append("--only_anomalies")
    run(cmd)


def print_summary(args, t0: float):
    elapsed = time.time() - t0
    reports = sorted(Path(args.out).glob("reports/reporte_*.xlsx"))
    banner("LISTO")
    print(f"  Tiempo total : {elapsed:.1f}s")
    if reports:
        print(f"  Ultimo reporte XLSX: {reports[-1]}")
        print(f"  Ultimo reporte CSV : {str(reports[-1]).replace('.xlsx','.csv')}")
    print(f"  Crops de anomalias : {args.out}/crops/")
    if args.template:
        print(f"  Calibracion        : output/calibration/")
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="E-14 Anomaly Detector — orquestador completo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_req = p.add_argument_group("Rutas principales")
    g_req.add_argument("--pdfs",      required=True,
                       help="Archivo .pdf o directorio con PDFs a analizar")
    g_req.add_argument("--dataset",   default=None,
                       help="Directorio dataset/ para entrenamiento (opcional si el modelo ya existe)")
    g_req.add_argument("--template",  default=None,
                       help="PDF plantilla para calibracion visual (opcional)")
    g_req.add_argument("--model",     default="models/isolation_forest.joblib")
    g_req.add_argument("--config",    default="config/pages_config.json")
    g_req.add_argument("--out",       default="output/")

    g_train = p.add_argument_group("Opciones de entrenamiento")
    g_train.add_argument("--epochs",  type=int,   default=40)
    g_train.add_argument("--batch",   type=int,   default=64)
    g_train.add_argument("--lr",      type=float, default=1e-3)
    g_train.add_argument("--val",     type=float, default=0.15)
    g_train.add_argument("--workers", type=int,   default=4)
    g_train.add_argument("--contamination", default="auto", help="Contaminacion para Isolation Forest")
    g_train.add_argument("--retrain", action="store_true",
                         help="Fuerza re-entrenamiento aunque exista el modelo")

    g_infer = p.add_argument_group("Opciones de inferencia")
    g_infer.add_argument("--threshold",      type=float, default=0.0)
    g_infer.add_argument("--entropy",        type=float, default=1.20)
    g_infer.add_argument("--margin",         type=float, default=0.25)
    g_infer.add_argument("--batch_infer",    type=int,   default=512)
    g_infer.add_argument("--only_anomalies", action="store_true")

    g_sys = p.add_argument_group("Sistema")
    g_sys.add_argument("--device", default="auto", help="cpu | cuda | auto")
    g_sys.add_argument("--zoom",   type=float, default=2.0, help="Zoom calibracion")
    g_sys.add_argument("--skip_install", action="store_true",
                       help="No ejecutar pip install (si las deps ya estan instaladas)")

    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    print()
    print("  ==========================================")
    print("  ===  E-14 ANOMALY DETECTOR — v1.0       ===")
    print("  ==========================================")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "reports"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "crops"),   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not args.skip_install:
        install_deps()

    if args.dataset or not Path(args.model).exists():
        if not args.dataset:
            print("\n[ERROR] No existe modelo y no se paso --dataset para entrenar.")
            sys.exit(1)
        do_train(args)

    do_calibrate(args)
    do_infer(args)
    print_summary(args, t0)


if __name__ == "__main__":
    main()
