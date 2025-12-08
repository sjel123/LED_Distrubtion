#!/bin/bash

PROJECT_DIR="/Users/sjelinsky/LEDMatrix/MonteCarlo"
VENV_DIR="/Users/sjelinsky/PythonEnv"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

source "$VENV_DIR/bin/activate"

exec gunicorn \
  -w 1 \
  --timeout 600 \
  -b 0.0.0.0:5030 \
  led_sim_pixel_flask:app \
  >> "$LOG_DIR/gunicorn.out.log" \
  2>> "$LOG_DIR/gunicorn.err.log"

