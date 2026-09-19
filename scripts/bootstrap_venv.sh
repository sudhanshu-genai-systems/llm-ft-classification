#!/usr/bin/env bash
# Isolated env for this repo — avoids conflicts with system LangChain / OpenAI installs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv"

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/pip" install -r "${ROOT}/requirements.txt"

echo ""
echo "Done. Activate with:"
echo "  source ${VENV}/bin/activate"
