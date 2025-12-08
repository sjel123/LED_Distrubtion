# `com.sjelinsky.ledmatrix` LaunchAgent

This LaunchAgent keeps `run_led_matrix_gunicorn.sh` alive so the LED pixel simulation UI stays reachable at `http://localhost:5030`. The plist reuses your Python virtualenv, runs within the MonteCarlo project directory, and redirects stdout/err into `logs/launchd.*.log` inside the repo.

## Setup

```sh
mkdir -p ~/Library/LaunchAgents                     # run once
cp com.sjelinsky.ledmatrix.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.sjelinsky.ledmatrix.plist
launchctl enable gui/$(id -u)/com.sjelinsky.ledmatrix
launchctl kickstart -k gui/$(id -u)/com.sjelinsky.ledmatrix
```

- `RunAtLoad` + `KeepAlive` ensure the job survives logouts and restarts automatically if it crashes.
- Logs land in `${HOME}/Library/Logs/ledmatrix-out.log` and `ledmatrix-err.log`; `tail -F` the files to watch activity.

## Control commands

To temporarily stop or totally unload the service (for upgrades):

```sh
launchctl disable gui/$(id -u)/com.sjelinsky.ledmatrix
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.sjelinsky.ledmatrix.plist
```

After editing `run_led_matrix_gunicorn.sh` or the plist, rerun the bootstrap/kickstart block above.

## Notes

- The plist hard-codes `/Users/sjelinsky/PythonEnv/bin/python`. Update `VENV_DIR` in `run_led_matrix_gunicorn.sh` or the plist if your virtual environment moves.
- Keep the working directory at `/Users/sjelinsky/LEDMatrix/MonteCarlo` so relative paths (templates, logs, scripts) continue to work.
- If you ever need to inspect the UI logs, open `logs/gunicorn.out.log` / `logs/gunicorn.err.log` inside the project directory.
