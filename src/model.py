import joblib
import numpy as np


class IsolationForestAnomalyDetector:
    """
    Wrapper que encapsula IsolationForest para deteccion de anomalias
    y RandomForestClassifier para clasificacion de digitos.
    """
    def __init__(self, isolation_forest, classifier, classes):
        self.isolation_forest = isolation_forest
        self.classifier = classifier
        self.classes = classes

    def predict(self, x_flat: np.ndarray):
        """
        x_flat: numpy array de forma (B, 1024) o (1024,) con pixeles normalizados.
        Retorna:
          - iforest_preds: array (B,) con 1 (inlier) y -1 (outlier/anomalia).
          - scores: array (B,) con puntuacion de anomalia (menor es mas anomalo).
          - probs: array (B, num_classes) con probabilidades de clase del RandomForest.
          - preds: array (B,) con clase predicha (indice entero).
        """
        if x_flat.ndim == 1:
            x_flat = x_flat.reshape(1, -1)

        # Scores de Isolation Forest: negativos son anomalias por defecto
        scores = self.isolation_forest.decision_function(x_flat)
        iforest_preds = self.isolation_forest.predict(x_flat)

        # Probabilidades y predicciones del clasificador
        probs = self.classifier.predict_proba(x_flat)
        preds = self.classifier.predict(x_flat)

        return iforest_preds, scores, probs, preds


def load_model(path: str, device: str = "cpu") -> IsolationForestAnomalyDetector:
    """Carga el modelo desde un archivo .joblib."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "isolation_forest" in payload:
        return IsolationForestAnomalyDetector(
            isolation_forest=payload["isolation_forest"],
            classifier=payload["classifier"],
            classes=payload.get("classes", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "dot"])
        )
    return payload


def save_model(model: IsolationForestAnomalyDetector, path: str, extra: dict = None):
    """Guarda el modelo en un archivo .joblib."""
    payload = {
        "isolation_forest": model.isolation_forest,
        "classifier": model.classifier,
        "classes": model.classes
    }
    if extra:
        payload.update(extra)
    joblib.dump(payload, path)
    print(f"Modelo guardado: {path}")