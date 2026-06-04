import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from pathlib import Path
from collections import Counter


CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "dot", "lines"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 32


def preprocess_digit_img(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte imagen BGR a tensor 32x32 normalizado en [0,1].
    Deja el digito como trazo oscuro sobre fondo blanco.
    """
    if img_bgr is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.astype(np.float32)
    gray = cv2.resize(gray.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    mean_val = gray.mean()
    if mean_val < 128:
        gray = 255 - gray

    gray = gray.astype(np.float32) / 255.0
    return gray


class DigitDataset(Dataset):
    """
    Carga imagenes desde dataset/<clase>/<imagen.{png,jpg,jpeg}>.
    Las 11 clases son: 0-9 y dot.
    """

    def __init__(self, root_dir: str, augment: bool = True):
        self.root_dir  = Path(root_dir)
        self.augment   = augment
        self.samples: list[tuple[Path, int]] = []

        for cls in CLASSES:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"[AVISO] Carpeta no encontrada: {cls_dir}")
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for f in cls_dir.glob(ext):
                    self.samples.append((f, CLASS_TO_IDX[cls]))

        if not self.samples:
            raise FileNotFoundError(f"No se encontraron imágenes en {root_dir}")

        counts = Counter(lbl for _, lbl in self.samples)
        print(f"Dataset: {len(self.samples)} muestras  |  clases: {dict(sorted(counts.items()))}")

        self._aug = T.Compose([
            T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=8),
            T.RandomPerspective(distortion_scale=0.12, p=0.4),
            T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.2)),
        ])

        self._class_counts = [counts.get(i, 1) for i in range(len(CLASSES))]

    def class_weights(self) -> list[float]:
        total = sum(self._class_counts)
        return [total / (len(CLASSES) * c) for c in self._class_counts]

    def weighted_sampler(self) -> WeightedRandomSampler:
        cw = self.class_weights()
        sample_weights = [cw[lbl] for _, lbl in self.samples]
        return WeightedRandomSampler(sample_weights, num_samples=len(self.samples), replacement=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        arr = preprocess_digit_img(img)

        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        if self.augment:
            tensor = self._aug(tensor)
            noise  = torch.randn_like(tensor) * 0.025
            tensor = (tensor + noise).clamp(0.0, 1.0)

        return tensor, label


def build_loaders(root_dir: str, val_split: float = 0.15,
                  batch_size: int = 64, num_workers: int = 4):
    import platform
    if platform.system() == "Windows":
        num_workers = 0

    full_ds = DigitDataset(root_dir, augment=False)
    n       = len(full_ds)
    n_val   = int(n * val_split)
    n_tr    = n - n_val

    g       = torch.Generator().manual_seed(42)
    all_idx = torch.randperm(n, generator=g).tolist()
    tr_idx  = all_idx[:n_tr]
    va_idx  = all_idx[n_tr:]

    tr_ds = DigitDataset(root_dir, augment=True)
    va_ds = DigitDataset(root_dir, augment=False)

    tr_subset = torch.utils.data.Subset(tr_ds, tr_idx)
    va_subset = torch.utils.data.Subset(va_ds, va_idx)

    cw             = tr_ds.class_weights()
    sample_weights = [cw[tr_ds.samples[i][1]] for i in tr_idx]
    sampler        = WeightedRandomSampler(sample_weights, num_samples=n_tr, replacement=True)

    tr_loader = DataLoader(
        tr_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    va_loader = DataLoader(
        va_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return tr_loader, va_loader