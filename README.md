# LED Matrix Monte Carlo Simulations

A pixel-at-a-time simulation server for driving a WLED 64×16 matrix. It hosts several statistical/visual simulations (Monte Carlo π, histograms, diffusion, random walk with a long-step variant, Lorenz attractor) plus idle animations, exposes a Flask control UI, and pushes updates directly to WLED (one pixel per change) to minimize bandwidth.

## Getting started

```bash
python3 led_sim_pixel_flask.py
```

The server listens on port `5030` and expects WLED at `192.168.1.181` by default; adjust `WLED_IP` if the device lives elsewhere.

## Configuration

- Use `templates/index.html` or the web UI to change mode, palettes, rotation, idle animation, timing parameters, and palette choices.
- Palette choices include `fire`, `plasma`, `viridis`, `turbo`, `neon`, `rainbow`, `aurora`, `random`, or a `single` color; the UI and CLI `--palette` flag accept the same names.
- Large constants (panel size, color stops, default palette) live near the top of `led_sim_pixel_flask.py` and its auxiliary modules (`led_sim_pixel_flask2.py`).
- Use `--config config.json` and `--palette`/`--idle-mode` CLI flags to preconfigure the web UI before start.

## Watching status

- `/status` returns current config/status JSON useful for monitoring or integrating with dashboards.
- Logs are written to `logs/launchd.*.log` when launched via `run_led_matrix_gunicorn.sh`.
