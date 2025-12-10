# LED Matrix Monte Carlo Simulations

Pixel-at-a-time simulation server for driving a WLED 64×16 matrix. It hosts several statistical/visual simulations (Monte Carlo π, histograms, diffusion, random walk with a long-step variant, a random pixel fill that lights unused pixels until the grid is full, an automated Snake '97 mode with apples, a classic Pac-Man chase that gobbles pellets while dodging ghosts, a worm trail that grows without overlapping itself, a row race that fills random rows until one stretches the full width, a column race that fills random columns until one spans the full height, multi-strip oscillator, wavefront pulse, Brownian cloud, reaction-diffusion textures, Lorenz attractor) plus idle animations, exposes a Flask control UI, and pushes updates directly to WLED (one pixel per change) to minimize bandwidth.

## Getting started

```bash
python3 led_sim_pixel_flask.py
```

The server listens on port `5030` and expects WLED at `192.168.1.181` by default; adjust `WLED_IP` if the device lives elsewhere.

## Configuration

- Use `templates/index.html` or the web UI to change mode, palettes, rotation, idle animation, timing parameters, and palette choices.
- The Pac-Man mode exposes a `pacman_ghosts` slider (1‑4 ghosts) so you can tune how many chasers run around the maze.
- Palette choices include `fire`, `plasma`, `viridis`, `turbo`, `neon`, `rainbow`, `aurora`, `random`, or a `single` color; the UI and CLI `--palette` flag accept the same names.
- The default rotation is 90° (logical 16×64) to optimize for tall panels; change to 0/180 if you prefer the 64×16 landscape grid.
- Large constants (panel size, color stops, default palette) live near the top of `led_sim_pixel_flask.py` and its auxiliary modules (`led_sim_pixel_flask2.py`).
- Use `--config config.json` and `--palette`/`--idle-mode` CLI flags to preconfigure the web UI before start.

## Watching status

- `/status` returns current config/status JSON useful for monitoring or integrating with dashboards.
- Logs are written to `logs/launchd.*.log` when launched via `run_led_matrix_gunicorn.sh`.
