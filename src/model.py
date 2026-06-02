import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))


class DigitCNN(nn.Module):
    """
    CNN liviana para clasificacion de digitos manuscritos en 11 clases (0-9, dot).
    Entrada: tensor [B, 1, 32, 32] en rango [0, 1].
    Salida:  logits [B, num_classes].
    """

    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.num_classes = num_classes

        self.layer1 = ConvBlock(1,  32, dropout=0.1)
        self.pool1  = nn.MaxPool2d(2)        # 16x16

        self.layer2 = ConvBlock(32, 64, dropout=0.1)
        self.pool2  = nn.MaxPool2d(2)        # 8x8

        self.layer3 = ConvBlock(64, 128, dropout=0.15)
        self.pool3  = nn.MaxPool2d(2)        # 4x4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return self.classifier(x)

    def predict(self, x: torch.Tensor):
        """
        Devuelve (clase, prob_max, distribucion_softmax).
        Acepta tensor sin dimensión de batch o con batch.
        """
        was_single = x.dim() == 3
        if was_single:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
        if was_single:
            probs = probs.squeeze(0)
        top_prob, top_cls = probs.max(dim=-1)
        return top_cls, top_prob, probs


def load_model(path: str, device: str = "cpu", num_classes: int = 11) -> DigitCNN:
    model = DigitCNN(num_classes=num_classes)
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_model(model: DigitCNN, path: str, extra: dict = None):
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"Modelo guardado: {path}")