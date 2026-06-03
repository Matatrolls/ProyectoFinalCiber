# E-14 Anomaly Detector

Aplicación de aprendizaje profundo para detectar anomalías en formularios E-14 escaneados (Actas de Escrutinio — Elecciones Colombia).

---

## Requisitos del sistema

| Dependencia | Versión mínima |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA (opcional) | 11.8+ |
| Poppler (solo PDFs reales) | cualquiera |

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd e14_detector

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

#    Linux:
apt install poppler-utils
#    Mac:
brew install poppler
#    Windows: descargar desde https://github.com/oschwartz10612/poppler-windows

#una vez instalado anadir a variables de entorno ej:
'C:\Users\Usuario\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin' 

```



---

## Estructura del dataset

El entrenamiento requiere una carpeta `dataset/` con esta estructura:

```
dataset/
├── 0/          ← imágenes del dígito 0
├── 1/          ← imágenes del dígito 1
├── ...
├── 9/          ← imágenes del dígito 9
└── dot/        ← imágenes de círculos vacíos / marcadores
```

Cada subcarpeta acepta archivos `.png`, `.jpg` o `.jpeg`.  
Se recomienda mínimo **200 imágenes por clase**.

---

## Flujo completo

```
dataset/ ──► train.py ──► models/isolation_forest.joblib
                                    │
pdfs/    ──────────────► infer.py ──┘──► output/reports/*.csv
                                         output/reports/*.xlsx
                                         output/crops/*.png
```

---

## Paso 1 — Entrenamiento

```bash
python train.py --dataset dataset/ --epochs 40
```

Parámetros disponibles:

| Parámetro | Default | Descripción |
|---|---|---|
| `--dataset` | `dataset/` | Ruta al directorio con imágenes |
| `--epochs` | `40` | Número de épocas |
| `--batch` | `64` | Tamaño de batch |
| `--lr` | `0.001` | Learning rate inicial |
| `--val` | `0.15` | Fracción de validación |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:0`, `auto` |
| `--workers` | `4` | Workers para DataLoader |
| `--out` | `models/` | Carpeta de salida del modelo |

El script guarda:
- `models/isolation_forest.joblib` — modelo Isolation Forest + Random Forest Classifier entrenado
- `models/train_history.json` — historial de métricas

---

## Paso 2 — Calibración de coordenadas (opcional pero recomendado)

Verifica visualmente que las cajas ROI coincidan con los campos del formulario **antes** de procesar miles de PDFs:

```bash
python calibrate.py --pdf ruta/a/plantilla.pdf --page 2
```

Genera imágenes anotadas en `output/calibration/` con las cajas de cada campo superpuestas sobre la página.

Si alguna caja no está alineada, edita los valores `bbox` en `config/pages_config.json`.

Parámetros:

| Parámetro | Default | Descripción |
|---|---|---|
| `--pdf` | — | PDF plantilla o archivo ZIP de páginas |
| `--page` | todas | Número de página a revisar (1-30) |
| `--zoom` | `2.0` | Factor de zoom para visualización |
| `--show` | — | Muestra ventana interactiva (requiere display) |
| `--out` | `output/calibration` | Directorio de salida |

---

## Paso 3 — Inferencia (detección de anomalías)

```bash
# Un solo archivo
python infer.py --pdfs E14_001.pdf

# Directorio completo
python infer.py --pdfs pdfs/

# Con umbrales personalizados
python infer.py --pdfs pdfs/ --threshold 0.90 --entropy 1.0 --margin 0.30

# Solo registrar anomalías (omitir campos OK)
python infer.py --pdfs pdfs/ --only_anomalies
```

Parámetros:

| Parámetro | Default | Descripción |
|---|---|---|
| `--pdfs` | — | Archivo `.pdf` o directorio |
| `--model` | `models/isolation_forest.joblib` | Ruta al modelo entrenado |
| `--config` | `config/pages_config.json` | Configuración de coordenadas |
| `--threshold` | `0.0` | Umbral de score de Isolation Forest (menor = anomalía) |
| `--entropy` | `1.20` | Entropía máxima permitida |
| `--margin` | `0.25` | Margen mínimo top1−top2 |
| `--batch` | `512` | Slots procesados por lote |
| `--device` | `auto` | Ignorado (mantenido por compatibilidad) |
| `--out` | `output/` | Directorio de salida |
| `--only_anomalies` | — | Solo reportar anomalías en CSV/XLSX |

---

## Formato de los reportes

Cada ejecución genera dos archivos en `output/reports/`:

### `reporte_YYYYMMDD_HHMMSS.csv`

```
archivo,pagina,seccion,lista,campo,candidato,columna,slot_digito,
prediccion,confianza,entropia,margen,top3,estado,motivo,crop
```

### `reporte_YYYYMMDD_HHMMSS.xlsx`

Igual al CSV más:
- Filas en **rojo** = anomalías
- Columna de imagen con el recorte del slot
- Hoja "Resumen" con conteo de anomalías por archivo

### Ejemplo de fila de anomalía

| Campo | Valor |
|---|---|
| archivo | E14_001.pdf |
| pagina | 10 |
| seccion | Circunscripcion Nacional |
| lista | Coalicion Cambio Radical - ALMA |
| campo | cand_009 |
| candidato | 9 |
| columna | left |
| slot_digito | 2 |
| prediccion | 8 |
| confianza | 41% |
| entropia | 1.847 |
| estado | **ANOMALIA** |
| motivo | baja_confianza (41%); entropia_alta (1.847) |

---

## Lógica de detección de anomalías

El sistema aplica **5 niveles de verificación** por cada slot de dígito:

| Nivel | Condición | Umbral configurable |
|---|---|---|
| Anotación Isolation Forest | `iforest_score < threshold` | `--threshold` (default `0.0`) |
| Entropía alta (suplementario) | `H(softmax) > max_entropy` | `--entropy` |
| Margen insuficiente (suplementario) | `top1 − top2 < margin` | `--margin` |
| Componentes múltiples | más de 1 componente conectado en el slot | `pages_config.json` |
| Tamaño anormal | bbox del componente muy grande o muy pequeño | `pages_config.json` |

Un slot con **cualquier** condición activa se marca `ANOMALIA`.

---

## Ajuste de coordenadas (`pages_config.json`)

La configuración describe cada página por plantilla:

```json
{
  "templates": {
    "nivelacion": {
      "fields": [
        { "name": "total_votantes_e11", "bbox": [383, 328, 527, 362], "num_digits": 3 }
      ]
    },
    "preferente_full": {
      "row0": { "bbox": [455, 384, 527, 412], ... },
      "table": {
        "y_start": 414, "row_height": 32,
        "columns": [
          { "col_id": "left",   "x1": 45,  "x2": 120 },
          { "col_id": "center", "x1": 238, "x2": 312 },
          { "col_id": "right",  "x1": 430, "x2": 510 }
        ]
      }
    }
  }
}
```

Coordenadas en píxeles para la imagen de referencia de **532 × 1568 px**.  
El sistema escala automáticamente a las dimensiones reales de cada PDF procesado.

---

## Formatos de PDF soportados

| Formato | Descripción |
|---|---|
| ZIP de JPEGs | Formato nativo de los formularios E-14 descargados del sistema CNE |
| PDF escaneado | Requiere Poppler instalado (`pdf2image`) |

El sistema detecta automáticamente el formato por la firma de bytes del archivo.

---

## Notas técnicas

- **Modelos de Aprendizaje**: Isolation Forest para detección de anomalías y Random Forest Classifier para clasificación de dígitos. Entrada 32×32 escala de grises aplanada (1024 características).
- **Augmentation**: rotación ±12°, perspectiva, blur gaussiano, ruido aleatorio, muestreo balanceado por clase.
- **Inferencia en lote**: todos los slots de un PDF se clasifican eficientemente en lote usando Scikit-Learn.
- **Escalado de coordenadas**: todas las coordenadas en `pages_config.json` se reescalan proporcionalmente al tamaño real de la imagen procesada.

---

##  Modo express — un solo comando

Tres scripts lanzan **todo el flujo automáticamente**: instalan dependencias, crean el entorno virtual, entrenan el modelo (si no existe) y generan los reportes.

### Linux / Mac

```bash
bash go.sh --dataset dataset/ --pdfs pdfs/
```

### Windows

```bat
go.bat --dataset dataset\ --pdfs pdfs\
```

### Python directo (cualquier SO)

```bash
python run.py --dataset dataset/ --pdfs pdfs/
```

El script detecta si el modelo ya está entrenado y **omite el entrenamiento** automáticamente en ejecuciones siguientes.

---

### Ejemplos de uso del comando único

```bash
# Primera vez: instala todo, entrena y analiza
bash go.sh --dataset dataset/ --pdfs pdfs/

# Segunda vez: solo analiza (modelo ya existe)
bash go.sh --pdfs pdfs/

# Con plantilla para verificar coordenadas antes de inferir
bash go.sh --dataset dataset/ --pdfs pdfs/ --template plantilla.pdf

# Umbral más estricto + solo reportar anomalías + GPU
bash go.sh --pdfs pdfs/ --threshold 0.90 --only_anomalies --device cuda

# Forzar re-entrenamiento con más épocas
bash go.sh --dataset dataset/ --pdfs pdfs/ --retrain --epochs 60

# Si las dependencias ya están instaladas (más rápido)
bash go.sh --pdfs pdfs/ --skip_install
```

---

### Parámetros del comando único

| Parámetro | Default | Descripción |
|---|---|---|
| `--pdfs` | — | **Requerido.** Archivo `.pdf` o directorio |
| `--dataset` | — | Dataset para entrenar. Omitir si el modelo ya existe |
| `--template` | — | PDF plantilla para calibración visual (opcional) |
| `--model` | `models/isolation_forest.joblib` | Ruta del modelo |
| `--retrain` | — | Fuerza re-entrenamiento aunque exista el modelo |
| `--epochs` | `40` | Ignorado (mantenido por compatibilidad) |
| `--batch` | `64` | Ignorado (mantenido por compatibilidad) |
| `--threshold` | `0.0` | Umbral de score de Isolation Forest para anomalía |
| `--entropy` | `1.20` | Entropía máxima permitida |
| `--margin` | `0.25` | Margen mínimo top1−top2 |
| `--only_anomalies` | — | Solo incluir anomalías en el reporte |
| `--device` | `auto` | Ignorado |
| `--skip_install` | — | Omite `pip install` (ejecuciones posteriores) |

---

### Fases del orquestador

```
[1] pip install -r requirements.txt   ← se puede omitir con --skip_install
[2] train.py                          ← se omite si el modelo ya existe
[3] calibrate.py                      ← solo si se pasa --template
[4] infer.py                          ← siempre
[5] Resumen con rutas de reportes
```
