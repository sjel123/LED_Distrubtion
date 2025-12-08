# User Guide

## Overview
This project simulates LED art for a WLED 64×16 panel by generating one pixel at a time and pushing only changed pixels via the WLED JSON API. It runs a Flask UI on `localhost:5030` to control the simulation parameters and idle behavior.

## Running the server
1. Ensure Python 3.11+ is installed and dependencies from `requirements.txt` (if present) are satisfied.
2. Run `python3 led_sim_pixel_flask.py`. The CLI accepts optional flags:
   - `--config <path>`: load overrides from a JSON file.
   - `--palette <fire|plasma|viridis|turbo|neon|rainbow|aurora|random|single>`: force the initial palette.
   - `--idle-mode <off|rainbow|noise|matrix|galaxy|sparkle>`: initial idle animation when paused.
3. Access `http://localhost:5030` to open the UI.
4. The UI controls mode, FPS, color pattern, rotation, idle animations, and distribution parameters for each simulation.

## Simulation controls
- **Mode** selects which algorithm drives the pixels (Monte Carlo π, histograms, random walk, a longer-step random walk variant, heat diffusion, Game of Life, stock GBM, Lorenz attractor).
- **Running** toggles between active simulation and configurable idle animation.
- **Rotation** swaps logical dimensions between 64×16 and 16×64; the code automatically remaps logical coordinates when rotation changes.
- **Timing** includes FPS, points-per-frame (samples per iteration), pixel reset threshold, and pause duration after a reset.
- **Distribution parameters** let you tune normal/Poisson histograms and Monte Carlo π domain size.
- **Palette/Color pattern** control palette selection (fire, plasma, viridis, turbo, neon, rainbow, aurora, random, or single-color), single-color overrides, and alternative gradients.
- **Idle animation** governs the fallback textures shown when simulations are paused.

## JSON status endpoint
Call `/status` to retrieve the latest `config` and `status` dictionaries; useful for monitoring or automating parameter sweeps.

## Deployment notes
- For persistent hosting, use `run_led_matrix_gunicorn.sh` (creates `logs/launchd.*.log`) or start the Flask app via `gunicorn` as shown in the history.
- Logs appear under `logs/launchd.out.log` and `logs/launchd.err.log` when using the launcher script.

## Launchctl service (macOS)
- Update `run_led_matrix_gunicorn.sh` with your Python virtualenv path (`VENV_DIR`) and any project-specific tweaks.
- Create a LaunchAgent plist (example `~/Library/LaunchAgents/com.ledmatrix.sim.plist`) that executes the shell script, for example via `ProgramArguments` pointing to `/bin/bash` and the script path.
- Load/start the service with
   ```bash
   launchctl load -w ~/Library/LaunchAgents/com.ledmatrix.sim.plist
   ```
- Check its status with `launchctl list | grep ledmatrix`. Logs stream into `logs/gunicorn.out.log` / `logs/gunicorn.err.log`.
- To stop or reload the service, run
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.ledmatrix.sim.plist
   ```
   then re-run `launchctl load -w ...` after editing the plist or script.

## Tips
- When rotating the display, the code recomputes logical width/height and rebuilds the simulation to match the new orientation.
- Auto-reset triggers after `pixel_reset_after` changed pixels to prevent the panel from saturating; the panel clears and the sim state resets.
- The UI posts to itself; clicking `Clear Matrix` sends a direct WLED clear command without changing other configuration values.
