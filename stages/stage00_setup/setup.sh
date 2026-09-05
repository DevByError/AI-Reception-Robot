#!/usr/bin/env bash
# Stage 00 - automated venv setup (Linux / macOS / Jetson - actual deployment device)
# Run from project root: bash stages/stage00_setup/setup.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "== Stage 00 setup (Linux/Jetson) =="
echo "Project root: $PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "-> Creating new venv at .venv ..."
    python3 -m venv .venv
else
    echo "-> .venv already exists, skipping creation."
fi

echo "-> Upgrading pip..."
./.venv/bin/python -m pip install --upgrade pip

echo "-> Installing requirements..."
./.venv/bin/python -m pip install -r stages/stage00_setup/requirements.txt

echo ""
echo "Done - venv ready and requirements installed."
echo "To activate the venv in any new terminal:"
echo "   source .venv/bin/activate"
echo ""
echo "Then run the hardware check:"
echo "   python stages/stage00_setup/test_hardware.py"
