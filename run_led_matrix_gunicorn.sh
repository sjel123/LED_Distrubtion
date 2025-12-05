#!/bin/bash

# Paths
PROJECT_DIR="/Users/sjelinsky/Documents/LEDMatrix/MonteCarlo"
VENV_DIR="/Users/sjelinsky/PythonEnv"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

# Activate your Python environment
source "$VENV_DIR/bin/activate"

# Start gunicorn:
# - 1 worker (one LED matrix)
# - 10 minute timeout so it doesn't kill the worker unnecessarily
exec gunicorn \
  -w 1 \
  --timeout 600 \
  -b 0.0.0.0:8000 \
  led_sim_pixel_flask:app \
  >> "$LOG_DIR/gunicorn.out.log" \
  2>> "$LOG_DIR/gunicorn.err.log"

