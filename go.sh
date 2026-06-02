#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  E-14 Anomaly Detector — lanzador Linux / Mac
#  Uso: bash go.sh --dataset dataset/ --pdfs pdfs/
#  Todos los args extra se pasan directamente a run.py
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" -c "import sys; print(sys.version_info.major*10+sys.version_info.minor)" 2>/dev/null)
        if [ "${VER:-0}" -ge 310 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Se requiere Python 3.10+. No se encontro ninguna version compatible."
    exit 1
fi

echo "  Usando: $PYTHON ($($PYTHON --version))"

if [ ! -d "venv" ]; then
    echo "  Creando entorno virtual..."
    "$PYTHON" -m venv venv
fi

source venv/bin/activate

"$PYTHON" run.py --skip_install "$@"
