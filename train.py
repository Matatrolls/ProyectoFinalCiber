"""
Entrena el clasificador CNN de digitos para el sistema E-14.

Uso:
    python train.py --dataset dataset/ --epochs 40 --batch 64 --lr 1e-3
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento CNN digitos E-14")
    p.add_argument("--dataset",  default="dataset/",      help="Ruta al directorio dataset/")
    p.add_argument("--out",      default="models/",       help="Directorio para guardar modelo")
    p.add_argument("--epochs",   type=int, default=40)
    p.add_argument("--batch",    type=int, default=64)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--val",      type=float, default=0.15, help="Fraccion validacion")
    p.add_argument("--device",   default="auto")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = total_acc = n = 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type if hasattr(device, "type") else "cpu",
                                enabled=scaler is not None):
            out  = model(imgs)
            loss = criterion(out, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs           = imgs.size(0)
        total_loss  += loss.item() * bs
        total_acc   += accuracy(out, labels) * bs
        n            += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        bs           = imgs.size(0)
        total_loss  += loss.item() * bs
        total_acc   += accuracy(out, labels) * bs
        n            += bs
    return total_loss / n, total_acc / n


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    from src.dataset import build_loaders, CLASSES
    from src.model   import DigitCNN, save_model

    tr_loader, va_loader = build_loaders(
        root_dir    = args.dataset,
        val_split   = args.val,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    model     = DigitCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler    = torch.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(args.out, exist_ok=True)
    best_acc  = 0.0
    history   = []

    print(f"\nIniciando entrenamiento — {args.epochs} epocas\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = eval_epoch(model, va_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "va_loss": va_loss, "va_acc": va_acc})

        flag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_path = os.path.join(args.out, "digit_cnn_best.pt")
            save_model(model, best_path, {"epoch": epoch, "val_acc": va_acc,
                                          "classes": CLASSES})
            flag = " ← mejor"

        print(
            f"Ep {epoch:03d}/{args.epochs} | "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f} | "
            f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}{flag}"
        )

    last_path = os.path.join(args.out, "digit_cnn_last.pt")
    save_model(model, last_path, {"epoch": args.epochs, "val_acc": va_acc,
                                  "classes": CLASSES})

    hist_path = os.path.join(args.out, "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEntrenamiento completo. Mejor val_acc: {best_acc:.4f}")
    print(f"Modelo guardado en: {args.out}")


if __name__ == "__main__":
    main()
