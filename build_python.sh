#!/bin/bash
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"
export PATH=$PATH:$HOME/.local/bin

uv run python -m PyInstaller --onefile opencv/calibrate.py
