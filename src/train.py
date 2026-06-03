"""
Entrena el detector de anomalías Isolation Forest y clasificador de dígitos para el sistema E-14.

Uso:
    python train.py --dataset dataset/ --out models/
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier

def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento Isolation Forest y Clasificador para digitos E-14")
    p.add_argument("--dataset",  default="dataset/",      help="Ruta al directorio dataset/")
    p.add_argument("--out",      default="models/",       help="Directorio para guardar modelo")
    p.add_argument("--epochs",   type=int, default=40,    help="Ignorado (mantenido por compatibilidad)")
    p.add_argument("--batch",    type=int, default=64,    help="Ignorado (mantenido por compatibilidad)")
    p.add_argument("--lr",       type=float, default=1e-3, help="Ignorado (mantenido por compatibilidad)")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--val",      type=float, default=0.15, help="Fraccion validacion")
    p.add_argument("--device",   default="auto",          help="Ignorado (mantenido por compatibilidad)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--contamination", default="auto",     help="Tasa de contaminación para Isolation Forest (float o 'auto')")
    return p.parse_args()


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Cargando dataset...")
    from src.dataset import build_loaders, CLASSES
    from src.model   import IsolationForestAnomalyDetector, save_model

    tr_loader, va_loader = build_loaders(
        root_dir    = args.dataset,
        val_split   = args.val,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    print("\nRecolectando datos de entrenamiento...")
    X_train = []
    y_train = []
    for imgs, labels in tr_loader:
        X_train.append(imgs.view(imgs.size(0), -1).numpy())
        y_train.append(labels.numpy())
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")

    print("\nEntrenando Isolation Forest (Detección de anomalías)...")
    contamination = args.contamination
    if contamination != "auto":
        try:
            contamination = float(contamination)
        except ValueError:
            contamination = "auto"

    isolation_forest = IsolationForest(contamination=contamination, random_state=args.seed, n_jobs=-1)
    isolation_forest.fit(X_train)

    print("Entrenando Random Forest Classifier (Clasificación de dígitos)...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    classifier.fit(X_train, y_train)

    # Crear el wrapper y guardar
    model = IsolationForestAnomalyDetector(isolation_forest, classifier, CLASSES)
    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "isolation_forest.joblib")
    save_model(model, best_path)

    # Validación
    X_val = []
    y_val = []
    for imgs, labels in va_loader:
        X_val.append(imgs.view(imgs.size(0), -1).numpy())
        y_val.append(labels.numpy())
    
    if len(X_val) > 0:
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        val_preds = classifier.predict(X_val)
        val_acc = np.mean(val_preds == y_val)
        
        val_iforest_preds = isolation_forest.predict(X_val)
        val_anomaly_rate = np.mean(val_iforest_preds == -1)

        print(f"\n[VALIDACIÓN] Accuracy del Clasificador: {val_acc:.4f}")
        print(f"[VALIDACIÓN] Proporción de anomalías detectadas: {val_anomaly_rate:.4f}")
        
        history = [
            {
                "epoch": 1,
                "tr_loss": 0.0,
                "tr_acc": float(np.mean(classifier.predict(X_train) == y_train)),
                "va_loss": 0.0,
                "va_acc": float(val_acc)
            }
        ]
    else:
        print("\n[VALIDACIÓN] No hay muestras en el set de validación.")
        history = []

    hist_path = os.path.join(args.out, "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEntrenamiento completo. Mejor val_acc del clasificador: {val_acc:.4f}" if len(X_val) > 0 else "\nEntrenamiento completo.")
    print(f"Modelo guardado en: {args.out}")


if __name__ == "__main__":
    main()