#!/usr/bin/env python3
"""
Pixel-only LED simulation server for WLED 16x64 matrix.

- Physical panel: 64x16 (W x H)
- Logical grid depends on rotation:
    - rotation = 0 or 180: 64 x 16
    - rotation = 90 or 270: 16 x 64

Simulation modes:
    - pi        : Monte Carlo π (quarter-circle in a square)
    - normal    : Normal distribution histogram
    - poisson   : Poisson distribution histogram
    - random_walk : Single-pixel random walk
    - oscillator  : Multi-strip oscillator with phase-shifted bands
    - life      : Conway's Game of Life
    - heat      : Heat diffusion / reaction-diffusion-like pattern
    - stock     : Stock price Monte Carlo (GBM sparkline)
    - lorenz    : Lorenz attractor projected onto grid

Other features:
    - Pixel build: one-pixel-at-a-time, sending only changed pixels
    - Auto-reset after N changed pixels (with pause and clear)
    - Color palettes: fire, plasma, viridis, turbo, neon, single
    - Color palettes: fire, plasma, viridis, turbo, neon, rainbow, aurora, single
    - Color patterns: data, random, grad_x, grad_y
    - Idle animations when simulations are paused:
        off, rainbow, noise, matrix, galaxy, sparkle

Run:
    python3 led_sim_pixel_flask.py [--config config.json] [--palette fire] [--idle-mode rainbow]
Then open:
    http://localhost:5000
"""

import math
import random
import time
import threading
import requests
import argparse
import json
import os
import sys
from collections import deque

from flask import Flask, request, render_template, jsonify

# ===== PHYSICAL WLED MATRIX SETTINGS =====
PHYS_WIDTH  = 64
PHYS_HEIGHT = 16
SEGMENT = 0

NLED = PHYS_WIDTH * PHYS_HEIGHT
WLED_IP = "192.168.1.181"   # <-- change if needed
URL  = f"http://{WLED_IP}/json/state"

# ===== GLOBAL STATE =====
config_lock = threading.Lock()
config = {
    # Modes:
    # "pi", "normal", "poisson", "random_walk", "life",
    # "heat", "stock", "lorenz", "oscillator"
    "mode": "pi",
    "fps": 15,
    "points_per_frame": 20,

    "pi_domain": 16,           # pixels, square for Monte Carlo π
    "mu": 0.0,                 # normal mean
    "sigma": 1.0,              # normal std dev
    "lam": 5.0,                # poisson lambda

    "pixel_reset_after": 800,  # changed pixels before reset (for main sims)
    "pixel_pause": 2.0,        # seconds to pause before clear+restart
    "worm_length": 16,         # length of the worm mode

    # Rotation angle in degrees: 0, 90, 180, or 270
    "rotation": 0,

    # Color palette / theme
    # "fire", "plasma", "viridis", "turbo", "neon", "rainbow", "aurora", "random", "single"
    "palette": "fire",

    # When palette == "single", use this hex color (rrggbb, no '#')
    "single_color": "ffffff",

    # How to interpret t for colors:
    # "data"   = use t from simulation/idle (default)
    # "random" = random color per pixel
    # "grad_x" = gradient left→right (logical x)
    # "grad_y" = gradient top→bottom (logical y)
    "color_pattern": "data",

    # Idle animation when running == False
    # "off", "rainbow", "noise", "matrix", "galaxy", "sparkle"
    "idle_mode": "rainbow",

    "running": True,
    "snake_length": 5,
    "snake_pause": 2.0,
    "pacman_ghosts": 3,
}

status = {
    "stats": "n=0",
    "last_error": "",
}

# ===== COLOR / PALETTE HELPERS =====

def clamp01(t):
    return max(0.0, min(1.0, t))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp01(t)
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )


def rgb_to_hex(c):
    r, g, b = c
    return f"{r:02x}{g:02x}{b:02x}"


def gradient_color(stops, t):
    """
    Given a list of RGB stops and t in [0,1], interpolate along the gradient.
    """
    t = clamp01(t)
    n = len(stops)
    if n == 0:
        return (255, 255, 255)
    if n == 1:
        return stops[0]
    pos = t * (n - 1)
    i = int(math.floor(pos))
    f = pos - i
    if i >= n - 1:
        return stops[-1]
    return lerp_color(stops[i], stops[i + 1], f)


# Palette definitions
FIRE_STOPS = [
    (0,   0,   0),
    (120, 0,   0),
    (255, 64,  0),
    (255, 160, 0),
    (255, 255, 255),
]

PLASMA_STOPS = [
    (20,  0,  70),
    (0,  40, 255),
    (0, 255, 200),
    (255, 255, 0),
    (255, 0, 130),
]

VIRIDIS_STOPS = [
    (68,   1,  84),
    (59,  82, 139),
    (33, 145, 140),
    (94, 201,  97),
    (253, 231, 37),
]

TURBO_STOPS = [
    (34,   9, 135),
    (68,  58, 171),
    (29, 118, 188),
    (50, 181, 110),
    (248, 231,  28),
    (245, 126,  21),
    (179,   4,  23),
]

NEON_STOPS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 128),
]

RAINBOW_STOPS = [
    (255, 0, 0),
    (255, 127, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
    (75, 0, 130),
    (148, 0, 211),
]

AURORA_STOPS = [
    (12, 9, 79),
    (2, 48, 71),
    (0, 99, 73),
    (0, 164, 148),
    (255, 200, 87),
    (245, 91, 59),
]


def palette_color(t, palette_name, single_color=None):
    """
    Map a scalar t in [0,1] to a hex color using the configured palette.
    If palette_name == "single", ignore t and return single_color.
    """
    p = (palette_name or "").lower()

    # Single-color mode
    if p == "single":
        col = (single_color or "ffffff").strip().lower()
        if col.startswith("#"):
            col = col[1:]
        if len(col) != 6:
            col = "ffffff"
        return col

    if p == "random":
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return rgb_to_hex((r, g, b))

    # Gradient palettes
    t = clamp01(t)
    if p == "fire":
        c = gradient_color(FIRE_STOPS, t)
    elif p == "plasma":
        c = gradient_color(PLASMA_STOPS, t)
    elif p == "viridis":
        c = gradient_color(VIRIDIS_STOPS, t)
    elif p == "turbo":
        c = gradient_color(TURBO_STOPS, t)
    elif p == "neon":
        c = gradient_color(NEON_STOPS, t)
    elif p == "rainbow":
        c = gradient_color(RAINBOW_STOPS, t)
    elif p == "aurora":
        c = gradient_color(AURORA_STOPS, t)
    else:
        # fallback: blue → cyan → yellow
        c = gradient_color(
            [(0, 0, 128), (0, 255, 255), (255, 255, 0)],
            t,
        )
    return rgb_to_hex(c)


def get_logical_dims(rotation):
    """
    Return (W_logical, H_logical) based on rotation angle.

    - rotation 0 or 180: logical = 64 x 16
    - rotation 90 or 270: logical = 16 x 64
    """
    if rotation in (0, 180):
        return PHYS_WIDTH, PHYS_HEIGHT
    else:
        return PHYS_HEIGHT, PHYS_WIDTH


def apply_color_pattern(t_base, lx, ly):
    """
    Central helper: given a base t (0..1) and logical coords (lx, ly),
    apply the configured color_pattern + palette to get a hex color.
    """
    with config_lock:
        palette_name   = config.get("palette", "fire")
        single_color   = config.get("single_color", "ffffff")
        color_pattern  = config.get("color_pattern", "data")
        rotation       = config.get("rotation", 0)

    log_W, log_H = get_logical_dims(rotation)
    pattern = (color_pattern or "data").lower()

    if pattern == "random":
        t = random.random()
    elif pattern == "grad_x":
        if log_W <= 1:
            t = t_base
        else:
            t = lx / float(log_W - 1)
    elif pattern == "grad_y":
        if log_H <= 1:
            t = t_base
        else:
            t = ly / float(log_H - 1)
    else:  # "data" or unknown
        t = t_base

    return palette_color(t, palette_name, single_color)


# ===== HTTP / LED HELPERS =====

def clear_matrix():
    """Set entire matrix to black once."""
    payload = {
        "on": True,
        "seg": [
            {
                "id": SEGMENT,
                "i": [0, NLED, "000000"]
            }
        ]
    }
    try:
        requests.post(URL, json=payload, timeout=2.0)
    except Exception as e:
        with config_lock:
            status["last_error"] = f"Error clearing matrix: {e}"


def logical_to_index(lx, ly):
    """
    Map logical (lx, ly) to physical WLED index according to rotation.

    Logical grid:
      - if rotation in (0, 180): size = (64 x 16)
      - if rotation in (90, 270): size = (16 x 64)

    Rotation is defined CCW:

      rotation = 0:
          (px, py) = (lx, ly)

      rotation = 90 (CCW):
          logical size: 16 x 64 (W_log x H_log)
          (px, py) = ( ly, PHYS_HEIGHT - 1 - lx )

      rotation = 180:
          (px, py) = ( PHYS_WIDTH - 1 - lx, PHYS_HEIGHT - 1 - ly )

      rotation = 270 (i.e. 90 CW):
          logical size: 16 x 64
          (px, py) = ( PHYS_WIDTH - 1 - ly, lx )
    """
    with config_lock:
        rotation = config.get("rotation", 0)

    if rotation in (0, 180):
        W_log, H_log = PHYS_WIDTH, PHYS_HEIGHT
    else:
        W_log, H_log = PHYS_HEIGHT, PHYS_WIDTH

    lx = max(0, min(W_log - 1, lx))
    ly = max(0, min(H_log - 1, ly))

    if rotation == 0:
        px, py = lx, ly
    elif rotation == 180:
        px = PHYS_WIDTH - 1 - lx
        py = PHYS_HEIGHT - 1 - ly
    elif rotation == 90:
        px = ly
        py = PHYS_HEIGHT - 1 - lx
    else:  # 270
        px = PHYS_WIDTH - 1 - ly
        py = lx

    px = max(0, min(PHYS_WIDTH  - 1, px))
    py = max(0, min(PHYS_HEIGHT - 1, py))

    return py * PHYS_WIDTH + px


def send_frame_batched(i_list):
    """
    Send one batched JSON update containing only changed pixels.

    i_list structure: [idx, 1, "rrggbb", idx2, 1, "rrggbb", ...]
    (no full-clear header; WLED merges changes onto current state)
    """
    if not i_list:
        return
    payload = {
        "on": True,
        "seg": [
            {
                "id": SEGMENT,
                "i": i_list
            }
        ]
    }
    try:
        requests.post(URL, json=payload, timeout=1.5)
    except Exception as e:
        with config_lock:
            status["last_error"] = f"Frame send error: {e}"


# ===== SIMULATION CLASSES (PIXEL-ONLY, USE LOGICAL W/H) =====

class MonteCarloPi:
    def __init__(self, domain_size, log_width, log_height):
        self.DOMAIN = min(domain_size, log_width, log_height)
        self.W = log_width
        self.H = log_height

        self.inside_hits  = [[0 for _ in range(self.DOMAIN)] for _ in range(self.DOMAIN)]
        self.outside_hits = [[0 for _ in range(self.DOMAIN)] for _ in range(self.DOMAIN)]
        self.total_points = 0
        self.inside_points = 0
        self.max_count = 0  # for per-cell brightness scaling

    def reset(self):
        self.inside_hits  = [[0 for _ in range(self.DOMAIN)] for _ in range(self.DOMAIN)]
        self.outside_hits = [[0 for _ in range(self.DOMAIN)] for _ in range(self.DOMAIN)]
        self.total_points = 0
        self.inside_points = 0
        self.max_count = 0

    def sample_pixel_step(self):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        r2 = x * x + y * y
        inside = r2 <= 1.0

        u = (x + 1.0) / 2.0
        v = (y + 1.0) / 2.0
        px = int(max(0, min(self.DOMAIN - 1, math.floor(u * self.DOMAIN))))
        py = int(max(0, min(self.DOMAIN - 1, math.floor(v * self.DOMAIN))))
        py = (self.DOMAIN - 1) - py

        if inside:
            self.inside_hits[py][px] += 1
            self.inside_points += 1
        else:
            self.outside_hits[py][px] += 1

        self.total_points += 1

        ins = self.inside_hits[py][px]
        outs = self.outside_hits[py][px]
        tot = ins + outs
        if tot > self.max_count:
            self.max_count = tot
        if self.max_count <= 0:
            self.max_count = 1

        density = tot / float(self.max_count)
        inside_frac = ins / float(tot) if tot > 0 else 0.0

        t_base = clamp01(0.4 * density + 0.6 * inside_frac)
        hex_color = apply_color_pattern(t_base, px, py)

        idx = logical_to_index(px, py)
        return idx, hex_color

    def stats_str(self):
        if self.total_points == 0:
            return "n=0"
        pi_est = 4.0 * self.inside_points / float(self.total_points)
        return f"n={self.total_points:8d}  inside={self.inside_points:8d}  π ≈ {pi_est:.6f}"


class NormalHistogram:
    def __init__(self, mu, sigma, log_width, log_height):
        self.mu = mu
        self.sigma = sigma
        self.W = log_width
        self.H = log_height

        self.x_min = mu - 3 * sigma
        self.x_max = mu + 3 * sigma
        self.bins = self.W
        self.counts = [0] * self.bins
        self.sample_count = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.next_y = [self.H - 1] * self.bins

    def reset(self):
        self.counts = [0] * self.bins
        self.sample_count = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.next_y = [self.H - 1] * self.bins

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        x = random.gauss(self.mu, self.sigma)
        self.sample_count += 1
        self.sum_x += x
        self.sum_x2 += x * x

        if x < self.x_min or x > self.x_max:
            return None

        u = (x - self.x_min) / (self.x_max - self.x_min)
        bin_idx = int(u * self.bins)
        if not (0 <= bin_idx < self.bins):
            return None

        self.counts[bin_idx] += 1

        y = self.next_y[bin_idx]
        if y < 0:
            y = 0
        else:
            self.next_y[bin_idx] -= 1

        t_base = bin_idx / float(max(1, self.bins - 1))
        hex_color = apply_color_pattern(t_base, bin_idx, y)

        idx = logical_to_index(bin_idx, y)
        return idx, hex_color

    def stats_str(self):
        if self.sample_count == 0:
            return "n=0"
        mean = self.sum_x / float(self.sample_count)
        var = (self.sum_x2 / float(self.sample_count)) - mean * mean
        return f"n={self.sample_count:8d}  mean≈{mean:6.3f}  var≈{var:6.3f}"


class PoissonHistogram:
    """
    Poisson histogram using the FULL LOGICAL WIDTH.
    """
    def __init__(self, lam, log_width, log_height):
        self.lam = lam
        self.W = log_width
        self.H = log_height

        if lam > 0:
            approx_max = lam + 6 * math.sqrt(lam)
        else:
            approx_max = 10
        self.k_max = max(5, int(approx_max))

        self.bins = max(1, self.W)
        self.counts = [0] * self.bins
        self.sample_count = 0
        self.sum_k = 0.0
        self.next_y = [self.H - 1] * self.bins

    def reset(self):
        self.counts = [0] * self.bins
        self.sample_count = 0
        self.sum_k = 0.0
        self.next_y = [self.H - 1] * self.bins

    def _sample_poisson(self):
        L = math.exp(-self.lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1

    def _k_to_bin(self, k):
        if self.bins <= 0 or self.k_max <= 0:
            return None
        if k < 0:
            return None
        kk = min(k, self.k_max)
        u = kk / float(self.k_max)
        bin_idx = int(u * (self.bins - 1))
        if 0 <= bin_idx < self.bins:
            return bin_idx
        return None

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        k = self._sample_poisson()
        self.sample_count += 1
        self.sum_k += k

        bin_idx = self._k_to_bin(k)
        if bin_idx is None:
            return None

        self.counts[bin_idx] += 1

        y = self.next_y[bin_idx]
        if y < 0:
            y = 0
        else:
            self.next_y[bin_idx] -= 1

        t_base = bin_idx / float(max(1, self.bins - 1))
        hex_color = apply_color_pattern(t_base, bin_idx, y)

        idx = logical_to_index(bin_idx, y)
        return idx, hex_color

    def stats_str(self):
        if self.sample_count == 0:
            return "n=0"
        mean = self.sum_k / float(self.sample_count)
        return f"n={self.sample_count:8d}  mean k≈{mean:6.3f}  λ={self.lam:5.2f}"

class RandomWalkSim:
    """
    Single-pixel random walk with trails in logical W x H.
    """
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.x = self.W // 2
        self.y = self.H // 2
        self.steps = 0

    def reset(self):
        self.x = self.W // 2
        self.y = self.H // 2
        self.steps = 0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.x = (self.x + dx) % self.W
        self.y = (self.y + dy) % self.H
        self.steps += 1

        t_base = (self.steps % 1000) / 1000.0
        hex_color = apply_color_pattern(t_base, self.x, self.y)

        idx = logical_to_index(self.x, self.y)
        return idx, hex_color

    def stats_str(self):
        return f"Random Walk: steps={self.steps:8d}"


class RandomWalkSim3:
    """
    Single-pixel random walk with trails in logical W x H.
    Steps can move between one and three pixels per update.
    All pixels in the jump are emitted so the entire path lights up.
    """
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.x = self.W // 2
        self.y = self.H // 2
        self.steps = 0

    def reset(self):
        self.x = self.W // 2
        self.y = self.H // 2
        self.steps = 0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        distance = random.randint(1, 3)

        path_pixels = []
        base_step = self.steps
        for i in range(1, distance + 1):
            px = (self.x + dx * i) % self.W
            py = (self.y + dy * i) % self.H
            step_num = base_step + i
            t_base = (step_num % 1000) / 1000.0
            hex_color = apply_color_pattern(t_base, px, py)
            idx = logical_to_index(px, py)
            path_pixels.append((idx, hex_color))

        self.x = (self.x + dx * distance) % self.W
        self.y = (self.y + dy * distance) % self.H
        self.steps += distance

        return path_pixels

    def stats_str(self):
        return f"Random Walk (longer steps): steps={self.steps:8d}"


class RowRaceSim:
    """Fill random rows until a row reaches full width and flash the winner."""
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.last_winner = None
        self.last_steps = 0
        self.pause_until = 0.0
        self._last_patch_skip_count = False
        self._start_new_race()

    def _start_new_race(self):
        self.counts = [0] * self.H
        self.steps = 0

    def reset(self):
        self._start_new_race()
        self.pause_until = 0.0
        self._last_patch_skip_count = False

    def sample_pixel_step(self):
        now = time.time()
        if self.pause_until and now < self.pause_until:
            return None
        if self.pause_until and now >= self.pause_until:
            self._start_new_race()
            self.pause_until = 0.0
            self._last_patch_skip_count = True
            return self._clear_patch()

        if self.W <= 0 or self.H <= 0:
            return None

        row = random.randrange(self.H)
        if self.counts[row] >= self.W:
            return None

        col = self.counts[row]
        self.counts[row] = min(self.W, col + 1)
        self.steps += 1

        if self.counts[row] >= self.W:
            self.last_winner = row
            winning_steps = self.steps
            winning_color = apply_color_pattern(0.5, self.W // 2, row)
            updates = [
                (logical_to_index(x, y), winning_color)
                for y in range(self.H)
                for x in range(self.W)
            ]
            self.last_steps = winning_steps
            self.pause_until = time.time() + 2.0
            self._last_patch_skip_count = True
            return updates

        denom = max(1, self.W - 1)
        t_base = col / float(denom)
        hex_color = apply_color_pattern(t_base, col, row)
        idx = logical_to_index(col, row)
        self._last_patch_skip_count = False
        return idx, hex_color

    def stats_str(self):
        if self.last_winner is None:
            return "Row race: racing..."
        return f"Row race: row {self.last_winner} won in {self.last_steps:5d} pixels"

    def _clear_patch(self):
        return [
            (logical_to_index(x, y), "000000")
            for y in range(self.H)
            for x in range(self.W)
        ]
class PacManSim:
    """Simple Pac-Man maze with pellets and roaming ghosts."""

    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    PACMAN_COLOR = "ffd700"
    GHOST_COLORS = ["ff4444", "ffbbff", "44ddff", "aaff44"]
    PELLET_COLOR = "c0d8ff"
    WALL_COLOR = "112244"

    def __init__(self, log_width, log_height, ghost_count=3):
        self.W = log_width
        self.H = log_height
        self.ghost_count = max(1, min(ghost_count, len(self.GHOST_COLORS)))
        self.walkable = []
        self.open_cells = []
        self._pacman_zone = []
        self._ghost_spawns = []
        self._pending_pellet_draw = []
        self._pending_reset = None
        self.level = 1
        self.score = 0
        self.steps = 0
        self._last_patch_skip_count = False
        self._build_maze()
        self.reset()

    def reset(self):
        self._build_maze()
        self.level = 1
        self.score = 0
        self.steps = 0
        self._pending_reset = None
        self._prepare_round(keep_level=False, keep_score=False)

    def sample_pixel_step(self):
        if self._pending_reset:
            self._apply_pending_reset()
        if self.pacman_pos is None or not self.open_cells:
            return None

        self.steps += 1
        pac_prev = self.pacman_pos
        ghost_prevs = [ghost["pos"] for ghost in self.ghosts]

        self._step_pacman()
        self._collect_pellet(self.pacman_pos)
        self._step_ghosts()
        ghost_news = [ghost["pos"] for ghost in self.ghosts]

        collision_cell = self._detect_collision(pac_prev, ghost_prevs, ghost_news)
        if collision_cell is not None:
            return self._handle_death(collision_cell)

        if not self.pellets:
            return self._handle_level_complete()

        updates = []
        update_cells = {pac_prev, self.pacman_pos}
        for prev, curr in zip(ghost_prevs, ghost_news):
            update_cells.add(prev)
            update_cells.add(curr)

        pellet_spot = self._draw_next_pellet()
        if pellet_spot:
            updates.append(pellet_spot)

        for x, y in update_cells:
            if not self._is_valid_coord(x, y):
                continue
            updates.append((logical_to_index(x, y), self._color_for(x, y)))

        return updates or None

    def stats_str(self):
        return f"Pac-Man L{self.level} score={self.score:5d} pellets={len(self.pellets):4d}"

    # ---------- Internal helpers ----------

    def _build_maze(self):
        self.walkable = [[True] * self.W for _ in range(self.H)]
        if self.W <= 0 or self.H <= 0:
            self.open_cells = []
            self._pacman_zone = []
            self._ghost_spawns = []
            return

        for x in range(self.W):
            self._set_walkable(x, 0, False)
            self._set_walkable(x, self.H - 1, False)
        for y in range(self.H):
            self._set_walkable(0, y, False)
            self._set_walkable(self.W - 1, y, False)

        if self.W > 4 and self.H > 4:
            for y in range(2, self.H - 2):
                for x in range(2, self.W - 2):
                    if ((x // 3) % 2 == 0 and (y % 6) not in (1, 4)) or (
                            (y // 3) % 2 == 0 and (x % 6) not in (1, 4)):
                        self.walkable[y][x] = False

        self._carve_center_zone()

        self.open_cells = [
            (x, y)
            for y in range(self.H)
            for x in range(self.W)
            if self.walkable[y][x]
        ]

        if not self.open_cells and self.W > 0 and self.H > 0:
            cx, cy = self.W // 2, self.H // 2
            self._set_walkable(cx, cy, True)
            self.open_cells = [(cx, cy)]

        self._define_zones()

    def _carve_center_zone(self):
        midx = self.W // 2
        midy = self.H // 2
        for dy in range(-2, 3):
            for dx in range(-3, 4):
                self._set_walkable(midx + dx, midy + dy, True)

    def _set_walkable(self, x, y, value):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.walkable[y][x] = value

    def _define_zones(self):
        if not self.open_cells:
            self._pacman_zone = []
            self._ghost_spawns = []
            return

        midx = self.W // 2
        midy = self.H // 2
        zone = []
        for dy in (-1, 0, 1):
            for dx in range(-2, 3):
                pos = self._clamp(midx + dx, midy + dy)
                if self._is_walkable(*pos):
                    zone.append(pos)
        if not zone:
            zone.append(self.open_cells[0])
        spawns = []
        offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        for dx, dy in offsets:
            pos = self._clamp(midx + dx, midy + dy)
            if self._is_walkable(*pos):
                spawns.append(pos)
        spawns = [pos for pos in spawns if pos not in zone]
        if not spawns:
            spawns.append(zone[0])

        self._pacman_zone = list(dict.fromkeys(zone))
        self._ghost_spawns = list(dict.fromkeys(spawns))

    def _clamp(self, x, y):
        return max(0, min(self.W - 1, x)), max(0, min(self.H - 1, y))

    def _is_walkable(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H and self.walkable[y][x]

    def _is_valid_coord(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def _prepare_round(self, keep_level=False, keep_score=False):
        self.steps = 0
        if not keep_score:
            self.score = 0
        if not keep_level:
            self.level = 1

        if not self.open_cells:
            self.pacman_pos = None
            self.ghosts = []
            self.pellets = set()
            self._pending_pellet_draw = []
            return

        self.pellets = set(self.open_cells)
        self.pacman_pos = self._choose_start_pos()
        self.pellets.discard(self.pacman_pos)

        spawn_pool = self._ghost_spawns or [self.pacman_pos]
        self.ghosts = []
        for i in range(self.ghost_count):
            spawn = spawn_pool[i % len(spawn_pool)]
            self.ghosts.append({"pos": spawn, "dir": random.choice(self.DIRECTIONS)})
            self.pellets.discard(spawn)

        self._pending_pellet_draw = list(self.pellets)
        random.shuffle(self._pending_pellet_draw)
        self.pacman_dir = random.choice(self.DIRECTIONS)

    def _choose_start_pos(self):
        if self._pacman_zone:
            return random.choice(self._pacman_zone)
        return random.choice(self.open_cells)

    def _step_pacman(self):
        if self.pacman_pos is None:
            return
        direction = self._choose_direction(self.pacman_pos, prefer=self.pacman_dir)
        if direction == (0, 0):
            return
        self.pacman_dir = direction
        next_pos = (self.pacman_pos[0] + direction[0], self.pacman_pos[1] + direction[1])
        if self._is_walkable(*next_pos):
            self.pacman_pos = next_pos

    def _step_ghosts(self):
        for ghost in self.ghosts:
            prev_pos = ghost["pos"]
            direction = self._choose_direction(prev_pos, prefer=ghost.get("dir"))
            if direction == (0, 0):
                direction = random.choice(self.DIRECTIONS)
            ghost["dir"] = direction
            target = (prev_pos[0] + direction[0], prev_pos[1] + direction[1])
            if not self._is_walkable(*target):
                options = [d for d in self.DIRECTIONS if self._is_walkable(prev_pos[0] + d[0], prev_pos[1] + d[1])]
                if options:
                    direction = random.choice(options)
                    ghost["dir"] = direction
                    target = (prev_pos[0] + direction[0], prev_pos[1] + direction[1])
                else:
                    target = prev_pos
            ghost["pos"] = target

    def _choose_direction(self, pos, prefer=None):
        candidates = []
        for dx, dy in self.DIRECTIONS:
            if self._is_walkable(pos[0] + dx, pos[1] + dy):
                candidates.append((dx, dy))
        if not candidates:
            return (0, 0)
        if prefer in candidates and random.random() < 0.7:
            return prefer
        if prefer:
            opposite = (-prefer[0], -prefer[1])
            filtered = [d for d in candidates if d != opposite]
            if filtered:
                return random.choice(filtered)
        return random.choice(candidates)

    def _collect_pellet(self, pos):
        if pos in self.pellets:
            self.pellets.remove(pos)
            self.score += 10

    def _draw_next_pellet(self):
        attempts = len(self._pending_pellet_draw)
        for _ in range(attempts):
            pos = self._pending_pellet_draw.pop()
            if pos not in self.pellets:
                continue
            if pos == self.pacman_pos or any(pos == ghost["pos"] for ghost in self.ghosts):
                continue
            return logical_to_index(pos[0], pos[1]), self.PELLET_COLOR
        return None

    def _color_for(self, x, y):
        if self.pacman_pos == (x, y):
            return self.PACMAN_COLOR
        for index, ghost in enumerate(self.ghosts):
            if ghost["pos"] == (x, y):
                return self.GHOST_COLORS[index % len(self.GHOST_COLORS)]
        if not self._is_walkable(x, y):
            return self.WALL_COLOR
        if (x, y) in self.pellets:
            return self.PELLET_COLOR
        return "000000"

    def _detect_collision(self, pac_prev, ghost_prevs, ghost_news):
        if pac_prev in ghost_prevs:
            return pac_prev
        for prev, news in zip(ghost_prevs, ghost_news):
            if news == self.pacman_pos:
                return self.pacman_pos
            if news == pac_prev and prev == self.pacman_pos:
                return pac_prev
        return None

    def _handle_death(self, center):
        self.score = max(0, self.score - 25)
        self._pending_reset = "death"
        self._last_patch_skip_count = True
        return self._death_flash(center)

    def _handle_level_complete(self):
        self.score += 50
        self.level += 1
        self._pending_reset = "level"
        self._last_patch_skip_count = True
        return self._level_flash(self.pacman_pos)

    def _apply_pending_reset(self):
        if not self._pending_reset:
            return
        if self._pending_reset == "death":
            self._prepare_round(keep_level=True, keep_score=True)
        elif self._pending_reset == "level":
            self._prepare_round(keep_level=True, keep_score=True)
        self._pending_reset = None

    def _death_flash(self, center):
        if center is None:
            return []
        updates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if abs(dx) + abs(dy) > 2:
                    continue
                x, y = self._wrap(center[0] + dx, center[1] + dy)
                if not self._is_valid_coord(x, y):
                    continue
                updates.append((logical_to_index(x, y), "ff2255"))
        return updates

    def _level_flash(self, center):
        if center is None:
            center = (self.W // 2, self.H // 2)
        updates = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 3:
                    continue
                x, y = self._wrap(center[0] + dx, center[1] + dy)
                if not self._is_valid_coord(x, y):
                    continue
                updates.append((logical_to_index(x, y), "59ff9f"))
        return updates

    def _wrap(self, x, y):
        if self.W <= 0 or self.H <= 0:
            return x, y
        return x % self.W, y % self.H



class ColumnRaceSim:
    """Fill random columns until a column reaches full height and flash the winner."""
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.last_winner = None
        self.last_steps = 0
        self.pause_until = 0.0
        self._last_patch_skip_count = False
        self._start_new_race()

    def _start_new_race(self):
        self.counts = [0] * self.W
        self.steps = 0

    def reset(self):
        self._start_new_race()
        self.pause_until = 0.0
        self._last_patch_skip_count = False

    def sample_pixel_step(self):
        now = time.time()
        if self.pause_until and now < self.pause_until:
            return None
        if self.pause_until and now >= self.pause_until:
            self._start_new_race()
            self.pause_until = 0.0
            self._last_patch_skip_count = True
            return self._clear_patch()

        if self.W <= 0 or self.H <= 0:
            return None

        col = random.randrange(self.W)
        if self.counts[col] >= self.H:
            return None

        row = self.counts[col]
        self.counts[col] = min(self.H, row + 1)
        self.steps += 1

        if self.counts[col] >= self.H:
            self.last_winner = col
            winning_steps = self.steps
            winning_color = apply_color_pattern(0.5, col, self.H // 2)
            updates = [
                (logical_to_index(x, y), winning_color)
                for y in range(self.H)
                for x in range(self.W)
            ]
            self.last_steps = winning_steps
            self.pause_until = time.time() + 2.0
            self._last_patch_skip_count = True
            return updates

        denom = max(1, self.H - 1)
        t_base = row / float(denom)
        hex_color = apply_color_pattern(t_base, col, row)
        idx = logical_to_index(col, row)
        self._last_patch_skip_count = False
        return idx, hex_color

    def stats_str(self):
        if self.last_winner is None:
            return "Column race: racing..."
        return f"Column race: column {self.last_winner} won in {self.last_steps:5d} pixels"

    def _clear_patch(self):
        return [
            (logical_to_index(x, y), "000000")
            for y in range(self.H)
            for x in range(self.W)
        ]


class RandomPixelSim:
    """Light random unlit pixels until the canvas fills, then clear and restart."""
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.available = []
        self.steps = 0
        self.pause_until = 0.0
        self._last_patch_skip_count = False
        self._start_new_round()

    def _start_new_round(self):
        self.available = [(x, y) for y in range(self.H) for x in range(self.W)]
        self.steps = 0

    def reset(self):
        self._start_new_round()
        self.pause_until = 0.0
        self._last_patch_skip_count = False

    def sample_pixel_step(self):
        if self.pause_until:
            now = time.time()
            if now < self.pause_until:
                return None
            self.pause_until = 0.0
            self._last_patch_skip_count = True
            self._start_new_round()
            return self._clear_patch()

        if not self.available:
            self._start_new_round()
            self.pause_until = time.time() + 0.5
            self._last_patch_skip_count = True
            return self._clear_patch()

        if self.W <= 0 or self.H <= 0:
            return None

        x, y = random.choice(self.available)
        self.available.remove((x, y))
        self.steps += 1

        t_base = random.random()
        hex_color = apply_color_pattern(t_base, x, y)
        idx = logical_to_index(x, y)
        self._last_patch_skip_count = False
        return idx, hex_color

    def stats_str(self):
        remaining = len(self.available)
        return f"Random pixels: {self.steps:4d} placed, {remaining:4d} left"

    def _clear_patch(self):
        return [
            (logical_to_index(x, y), "000000")
            for y in range(self.H)
            for x in range(self.W)
        ]


class WormSim:
    """Move a fixed-length worm without self-overlaps, wrapping the grid like Snake."""
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, log_width, log_height, length=16):
        self.W = log_width
        self.H = log_height
        self.length = max(1, length)
        self.positions = deque()
        self.occupied = set()
        self.current_direction = random.choice(self.DIRECTIONS)
        self.steps = 0
        self.pause_until = 0.0
        self._clear_after_pause = False
        self._last_patch_skip_count = False
        self._start_new_worm()

    def _start_new_worm(self):
        self.positions.clear()
        self.occupied.clear()
        mid_x = self.W // 2
        mid_y = self.H // 2
        for offset in range(self.length):
            x = (mid_x - offset) % max(1, self.W)
            y = mid_y % max(1, self.H)
            self.positions.append((x, y))
        self.occupied.update(self.positions)
        self.current_direction = random.choice(self.DIRECTIONS)
        self.steps = 0

    def reset(self):
        self._start_new_worm()

    def _wrapped(self, x, y):
        return x % max(1, self.W), y % max(1, self.H)

    def sample_pixel_step(self):
        now = time.time()
        if self.pause_until:
            if now < self.pause_until:
                return None
            self.pause_until = 0.0
            if self._clear_after_pause:
                self._clear_after_pause = False
                self._start_new_worm()
                self._last_patch_skip_count = True
                return self._clear_patch()

        if self.W <= 0 or self.H <= 0 or not self.positions:
            return None

        head_x, head_y = self.positions[0]
        tail = self.positions[-1]
        neck = self.positions[1] if len(self.positions) > 1 else None
        candidates = self.DIRECTIONS.copy()
        random.shuffle(candidates)
        next_pos = None
        for dx, dy in candidates:
            if neck and (head_x + dx, head_y + dy) == neck:
                continue
            nx, ny = self._wrapped(head_x + dx, head_y + dy)
            if (nx, ny) == tail or (nx, ny) not in self.occupied:
                next_pos = (nx, ny)
                self.current_direction = (dx, dy)
                break
        if next_pos is None:
            explosion = self._mini_explosion(head_x, head_y)
            self.pause_until = time.time() + 2.0
            self._clear_after_pause = True
            self._last_patch_skip_count = True
            return explosion

        tail = self.positions.pop()
        self.occupied.remove(tail)
        self.positions.appendleft(next_pos)
        self.occupied.add(next_pos)
        self.steps += 1

        t_base = (self.steps % 256) / 255.0
        hex_color = apply_color_pattern(t_base, next_pos[0], next_pos[1])
        head_idx = logical_to_index(next_pos[0], next_pos[1])
        tail_idx = logical_to_index(tail[0], tail[1])
        self._last_patch_skip_count = False
        return [
            (tail_idx, "000000"),
            (head_idx, hex_color),
        ]

    def stats_str(self):
        return f"Worm: length={self.length}, steps={self.steps:5d}"

    def _mini_explosion(self, cx, cy):
        with config_lock:
            palette = config.get("palette", "fire")
            single_color = config.get("single_color", "ffffff")
        color = palette_color(0.8, palette, single_color)
        updates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = self._wrapped(cx + dx, cy + dy)
                idx = logical_to_index(nx, ny)
                updates.append((idx, color))
        return updates


class SnakeSim:
    """Automated Snake '97-style game moving a growing snake toward fruit."""
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, log_width, log_height, length=5, pause=2.0):
        self.W = log_width
        self.H = log_height
        self.length = max(2, length)
        self.pause = max(0.0, pause)
        self.positions = deque()
        self.occupied = set()
        self.direction = random.choice(self.DIRECTIONS)
        self.fruit = None
        self._fruit_dirty = False
        self.steps = 0
        self.apples = 0
        self.pause_until = 0.0
        self._clear_after_pause = False
        self._last_patch_skip_count = False
        self._start_new_game()

    def _start_new_game(self):
        self.positions.clear()
        self.occupied.clear()
        mid_x = self.W // 2
        mid_y = self.H // 2
        for offset in range(self.length):
            x = (mid_x - offset) % max(1, self.W)
            y = mid_y % max(1, self.H)
            self.positions.append((x, y))
        self.occupied.update(self.positions)
        self.direction = random.choice(self.DIRECTIONS)
        self.steps = 0
        self.apples = 0
        self._place_fruit()
        self._clear_after_pause = False
        self._last_patch_skip_count = False

    def reset(self):
        self._start_new_game()

    def _wrapped(self, x, y):
        return x % max(1, self.W), y % max(1, self.H)

    def _place_fruit(self):
        choices = [
            (x, y)
            for y in range(self.H)
            for x in range(self.W)
            if (x, y) not in self.occupied
        ]
        if not choices:
            self.fruit = None
            self._fruit_dirty = False
            return
        self.fruit = random.choice(choices)
        self._fruit_dirty = True

    def _fruit_color(self):
        with config_lock:
            palette = config.get("palette", "fire")
            single_color = config.get("single_color", "ffffff")
        return palette_color(0.95, palette, single_color)

    def _clear_patch(self):
        return [
            (logical_to_index(x, y), "000000")
            for y in range(self.H)
            for x in range(self.W)
        ]

    def _manhattan_wrap(self, px, py, fx, fy):
        dx = min((fx - px) % self.W, (px - fx) % self.W)
        dy = min((fy - py) % self.H, (py - fy) % self.H)
        return dx + dy

    def sample_pixel_step(self):
        now = time.time()
        if self.pause_until:
            if now < self.pause_until:
                return None
            self.pause_until = 0.0
            if self._clear_after_pause:
                self._clear_after_pause = False
                self._start_new_game()
                self._last_patch_skip_count = True
                return self._clear_patch()

        if self.W <= 0 or self.H <= 0 or not self.positions:
            return None

        head_x, head_y = self.positions[0]
        tail = self.positions[-1]
        neck = self.positions[1] if len(self.positions) > 1 else None
        fx, fy = self.fruit if self.fruit else (head_x, head_y)

        candidates = []
        for dx, dy in self.DIRECTIONS:
            if neck and (head_x + dx, head_y + dy) == neck:
                continue
            nx, ny = self._wrapped(head_x + dx, head_y + dy)
            blocked = (nx, ny) in self.occupied and (nx, ny) != tail
            dist = self._manhattan_wrap(nx, ny, fx, fy)
            candidates.append(((dx, dy), blocked, dist))
        candidates.sort(key=lambda item: (item[1], item[2]))
        next_pos = None
        for (dx, dy), blocked, _ in candidates:
            if blocked:
                continue
            nx, ny = self._wrapped(head_x + dx, head_y + dy)
            next_pos = (nx, ny)
            self.direction = (dx, dy)
            break

        if next_pos is None:
            explosion = self._mini_explosion(head_x, head_y)
            self.pause_until = time.time() + self.pause
            self._clear_after_pause = True
            self._last_patch_skip_count = True
            return explosion

        growing = self.fruit is not None and next_pos == self.fruit
        if not growing:
            tail = self.positions.pop()
            self.occupied.remove(tail)
        else:
            self.apples += 1
            self._place_fruit()

        self.positions.appendleft(next_pos)
        self.occupied.add(next_pos)
        self.steps += 1

        updates = []
        if not growing:
            tail_idx = logical_to_index(tail[0], tail[1])
            updates.append((tail_idx, "000000"))
        head_idx = logical_to_index(next_pos[0], next_pos[1])
        snake_color = apply_color_pattern((self.steps % 256) / 255.0, next_pos[0], next_pos[1])
        updates.append((head_idx, snake_color))
        if self._fruit_dirty and self.fruit:
            fruit_idx = logical_to_index(self.fruit[0], self.fruit[1])
            updates.append((fruit_idx, self._fruit_color()))
            self._fruit_dirty = False

        self._last_patch_skip_count = False
        return updates

    def stats_str(self):
        return f"Snake '97: length={len(self.positions):4d}, apples={self.apples:3d}"

    def _mini_explosion(self, cx, cy):
        with config_lock:
            palette = config.get("palette", "fire")
            single_color = config.get("single_color", "ffffff")
        color = palette_color(0.8, palette, single_color)
        updates = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = self._wrapped(cx + dx, cy + dy)
                idx = logical_to_index(nx, ny)
                updates.append((idx, color))
        return updates


    def _clear_patch(self):
        return [
            (logical_to_index(x, y), "000000")
            for y in range(self.H)
            for x in range(self.W)
        ]
class MultiStripOscillator:
    """Oscillating multi-strip bands that pulse with independent phases."""
    def __init__(self, log_width, log_height, band_count=8):
        self.W = log_width
        self.H = log_height
        self.band_count = max(1, min(band_count, self.H or 1))
        self.phases = [random.uniform(0, 2 * math.pi) for _ in range(self.band_count)]
        self.speeds = [0.01 + random.random() * 0.03 for _ in range(self.band_count)]
        self.index = 0
        self.pixel_count = max(1, self.W * self.H)

    def reset(self):
        self.phases = [random.uniform(0, 2 * math.pi) for _ in range(self.band_count)]
        self.index = 0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        lx = self.index % self.W
        ly = (self.index // self.W) % self.H

        band = min(self.band_count - 1, int(ly * self.band_count / max(1, self.H)))
        phase = self.phases[band]
        brightness = 0.3 + 0.7 * (0.5 * (1 + math.sin(phase)))
        self.phases[band] += self.speeds[band]

        hex_color = apply_color_pattern(brightness, lx, ly)
        idx = logical_to_index(lx, ly)

        self.index = (self.index + 1) % self.pixel_count
        return idx, hex_color

    def stats_str(self):
        return f"Oscillator: {self.band_count} bands"


class GameOfLifeSim:
    """
    Conway's Game of Life on a logical W x H grid.
    """
    def __init__(self, log_width, log_height, alive_prob=0.3):
        self.W = log_width
        self.H = log_height
        self.alive_prob = alive_prob
        self.generation = 0
        self.grid = [[False for _ in range(self.W)] for _ in range(self.H)]
        self.prev_grid = [[False for _ in range(self.W)] for _ in range(self.H)]
        self.update_queue = []
        self._random_seed()

    def _random_seed(self):
        for y in range(self.H):
            for x in range(self.W):
                self.grid[y][x] = (random.random() < self.alive_prob)
        self.prev_grid = [[False for _ in range(self.W)] for _ in range(self.H)]
        self.update_queue = []
        self.generation = 0
        for y in range(self.H):
            for x in range(self.W):
                if self.grid[y][x] != self.prev_grid[y][x]:
                    self.update_queue.append((x, y, self.grid[y][x]))
        self._sync_prev()

    def _sync_prev(self):
        for y in range(self.H):
            for x in range(self.W):
                self.prev_grid[y][x] = self.grid[y][x]

    def reset(self):
        self._random_seed()

    def _step_generation(self):
        new_grid = [[False for _ in range(self.W)] for _ in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                neighbors = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = (y + dy) % self.H
                        nx = (x + dx) % self.W
                        if self.grid[ny][nx]:
                            neighbors += 1
                if self.grid[y][x]:
                    new_grid[y][x] = (neighbors == 2 or neighbors == 3)
                else:
                    new_grid[y][x] = (neighbors == 3)
        self.update_queue = []
        for y in range(self.H):
            for x in range(self.W):
                if new_grid[y][x] != self.grid[y][x]:
                    self.update_queue.append((x, y, new_grid[y][x]))
        self.grid = new_grid
        self._sync_prev()
        self.generation += 1
        if not self.update_queue:
            self._random_seed()

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        if not self.update_queue:
            self._step_generation()
        if not self.update_queue:
            return None
        x, y, alive = self.update_queue.pop()
        if alive:
            t_base = (self.generation % 256) / 255.0
            hex_color = apply_color_pattern(t_base, x, y)
        else:
            hex_color = "000000"

        idx = logical_to_index(x, y)
        return idx, hex_color

    def stats_str(self):
        return f"Game of Life: gen={self.generation:6d}"


class HeatDiffusionSim:
    """
    Simple heat diffusion / reaction-diffusion-like pattern on logical W x H.

    We keep a scalar field u[y][x] in [0,1], and iteratively diffuse it,
    with a bit of random injection.
    """
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.u = [[random.random() * 0.2 for _ in range(self.W)] for _ in range(self.H)]
        self.index = 0
        self.alpha = 0.5    # diffusion rate
        self.decay = 0.01   # slow decay
        self.inject_prob = 0.001

    def reset(self):
        self.u = [[random.random() * 0.2 for _ in range(self.W)] for _ in range(self.H)]
        self.index = 0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        lx = self.index % self.W
        ly = (self.index // self.W) % self.H
        self.index = (self.index + 1) % (self.W * self.H)

        # 4-neighbor diffusion
        u0 = self.u[ly][lx]
        neighbors = 0.0
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = (lx + dx) % self.W
            ny = (ly + dy) % self.H
            neighbors += self.u[ny][nx]
            count += 1
        if count > 0:
            avg = neighbors / float(count)
        else:
            avg = u0

        u_new = (1.0 - self.alpha) * u0 + self.alpha * avg
        u_new *= (1.0 - self.decay)

        # occasional random injection
        if random.random() < self.inject_prob:
            u_new += random.random() * 0.5
        u_new = clamp01(u_new)

        self.u[ly][lx] = u_new

        t_base = u_new
        hex_color = apply_color_pattern(t_base, lx, ly)

        idx = logical_to_index(lx, ly)
        return idx, hex_color

    def stats_str(self):
        return "Heat diffusion"


class StockGBMSim:
    """
    Stock price Monte Carlo (Geometric Brownian Motion) sparkline.

    - Time axis along logical X.
    - Multiple paths stacked vertically.
    - Each new column advances time; paths are updated by GBM.
    """
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height
        self.n_paths = max(1, self.H // 4)
        self.mu = 0.05
        self.sigma = 0.3
        self.dt = 1.0

        self.paths = [1.0 for _ in range(self.n_paths)]
        self.s_min = 1.0
        self.s_max = 1.0
        self.x = 0
        self.current_path = 0
        self.steps = 0

    def reset(self):
        self.paths = [1.0 for _ in range(self.n_paths)]
        self.s_min = 1.0
        self.s_max = 1.0
        self.x = 0
        self.current_path = 0
        self.steps = 0

    def _advance_price(self, i):
        s = self.paths[i]
        z = random.gauss(0.0, 1.0)
        drift = (self.mu - 0.5 * self.sigma * self.sigma) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * z
        s_new = s * math.exp(drift + diffusion)
        if s_new <= 0:
            s_new = 1e-4
        self.paths[i] = s_new
        if self.steps == 0 and i == 0:
            self.s_min = s_new
            self.s_max = s_new
        else:
            self.s_min = min(self.s_min, s_new)
            self.s_max = max(self.s_max, s_new)
        return s_new

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        if self.n_paths <= 0:
            return None

        i = self.current_path

        # new column when starting path 0
        if i == 0:
            if self.steps > 0:
                self.x = (self.x + 1) % self.W

        s = self._advance_price(i)

        if self.s_max <= self.s_min:
            t_price = 0.5
        else:
            t_price = (s - self.s_min) / float(self.s_max - self.s_min)
        t_price = clamp01(t_price)

        # Map price to vertical position (higher price -> top)
        ly = self.H - 1 - int(t_price * (self.H - 1))

        # Spread paths vertically if possible
        if self.n_paths > 1:
            # path_index in [0, n_paths-1]
            band_height = self.H / float(self.n_paths)
            base = int(i * band_height)
            ly = max(0, min(self.H - 1, base + ly % max(1, int(band_height))))
        ly = max(0, min(self.H - 1, ly))

        # Color by path index
        if self.n_paths > 1:
            t_base = i / float(self.n_paths - 1)
        else:
            t_base = t_price

        lx = self.x
        hex_color = apply_color_pattern(t_base, lx, ly)

        idx = logical_to_index(lx, ly)

        self.current_path = (self.current_path + 1) % self.n_paths
        if self.current_path == 0:
            self.steps += 1

        return idx, hex_color

    def stats_str(self):
        return f"Stock GBM: paths={self.n_paths}, steps={self.steps}"


class LorenzSim:
    """
    Lorenz attractor projected to logical W x H.
    """
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height

        # Lorenz parameters
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = 0.01

        # State
        self.x = 1.0
        self.y = 1.0
        self.z = 1.0

        self.min_x = self.x
        self.max_x = self.x
        self.min_y = self.y
        self.max_y = self.y
        self.min_z = self.z
        self.max_z = self.z

        self.steps = 0

    def reset(self):
        self.x = 1.0
        self.y = 1.0
        self.z = 1.0
        self.min_x = self.x
        self.max_x = self.x
        self.min_y = self.y
        self.max_y = self.y
        self.min_z = self.z
        self.max_z = self.z
        self.steps = 0

    def _step(self):
        # simple Euler integration
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt

        self.min_x = min(self.min_x, self.x)
        self.max_x = max(self.max_x, self.x)
        self.min_y = min(self.min_y, self.y)
        self.max_y = max(self.max_y, self.y)
        self.min_z = min(self.min_z, self.z)
        self.max_z = max(self.max_z, self.z)

        self.steps += 1

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        self._step()

        if self.max_x <= self.min_x:
            u = 0.5
        else:
            u = (self.x - self.min_x) / float(self.max_x - self.min_x)
        if self.max_z <= self.min_z:
            v = 0.5
        else:
            v = (self.z - self.min_z) / float(self.max_z - self.min_z)

        u = clamp01(u)
        v = clamp01(v)

        lx = int(u * (self.W - 1))
        ly = int(v * (self.H - 1))

        # Color using y coordinate
        if self.max_y <= self.min_y:
            t_base = 0.5
        else:
            t_base = (self.y - self.min_y) / float(self.max_y - self.min_y)
        t_base = clamp01(t_base)

        hex_color = apply_color_pattern(t_base, lx, ly)

        idx = logical_to_index(lx, ly)
        return idx, hex_color

    def stats_str(self):
        return f"Lorenz: steps={self.steps}"


# ===== IDLE ANIMATION CLASSES =====

class IdleBase:
    def __init__(self, log_width, log_height):
        self.W = log_width
        self.H = log_height


class IdleRainbow(IdleBase):
    def __init__(self, log_width, log_height):
        super().__init__(log_width, log_height)
        self.index = 0
        self.phase = 0.0

    def reset(self):
        self.index = 0
        self.phase = 0.0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        lx = self.index % self.W
        ly = (self.index // self.W) % self.H
        self.index = (self.index + 1) % (self.W * self.H)

        # Base swirl coordinate
        pos = (lx / float(max(1, self.W - 1)) + self.phase) % 1.0
        self.phase = (self.phase + 0.0005) % 1.0

        t_base = pos
        hex_color = apply_color_pattern(t_base, lx, ly)

        idx = logical_to_index(lx, ly)
        return idx, hex_color


class IdleNoise(IdleBase):
    def __init__(self, log_width, log_height):
        super().__init__(log_width, log_height)
        self.index = 0
        self.z = random.random()

    def reset(self):
        self.index = 0
        self.z = random.random()

    def _noise(self, x, y, z):
        return (math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453) % 1.0

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        lx = self.index % self.W
        ly = (self.index // self.W) % self.H
        self.index = (self.index + 1) % (self.W * self.H)

        self.z += 0.01
        v = self._noise(lx, ly, self.z)
        t_base = clamp01(v)

        hex_color = apply_color_pattern(t_base, lx, ly)

        idx = logical_to_index(lx, ly)
        return idx, hex_color


class IdleMatrixRain(IdleBase):
    def __init__(self, log_width, log_height):
        super().__init__(log_width, log_height)
        self.heads = [random.randint(-self.H, self.H) for _ in range(self.W)]

    def reset(self):
        self.heads = [random.randint(-self.H, self.H) for _ in range(self.W)]

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None

        x = random.randint(0, self.W - 1)
        head = self.heads[x]

        tail_y = head - 4
        if 0 <= tail_y < self.H:
            idx_tail = logical_to_index(x, tail_y)
            self.heads[x] = head + 1
            return idx_tail, "000000"

        y = head % (self.H + 10)
        if y >= self.H:
            self.heads[x] = head + 1
            return None

        t_base = clamp01(0.4 + 0.2 * random.random())
        hex_color = apply_color_pattern(t_base, x, y)

        idx = logical_to_index(x, y)
        self.heads[x] = head + 1

        if self.heads[x] > self.H + 10 and random.random() < 0.3:
            self.heads[x] = random.randint(-self.H, 0)

        return idx, hex_color


class IdleGalaxy(IdleBase):
    def __init__(self, log_width, log_height):
        super().__init__(log_width, log_height)
        self.stars = []
        for _ in range((self.W * self.H) // 16):
            self.stars.append({
                "x": random.randint(0, self.W - 1),
                "y": random.randint(0, self.H - 1),
                "phase": random.random(),
                "speed": 0.01 + 0.03 * random.random(),
            })

    def reset(self):
        self.__init__(self.W, self.H)

    def sample_pixel_step(self):
        if not self.stars:
            return None
        s = random.choice(self.stars)
        s["phase"] = (s["phase"] + s["speed"]) % 1.0
        brightness = 0.2 + 0.8 * (0.5 * (1 + math.sin(2 * math.pi * s["phase"])))
        t_base = clamp01(brightness)

        hex_color = apply_color_pattern(t_base, s["x"], s["y"])

        idx = logical_to_index(s["x"], s["y"])
        return idx, hex_color


class IdleSparkle(IdleBase):
    def __init__(self, log_width, log_height):
        super().__init__(log_width, log_height)

    def reset(self):
        pass

    def sample_pixel_step(self):
        if self.W <= 0 or self.H <= 0:
            return None
        lx = random.randint(0, self.W - 1)
        ly = random.randint(0, self.H - 1)
        t_base = random.random()
        hex_color = apply_color_pattern(t_base, lx, ly)
        idx = logical_to_index(lx, ly)
        return idx, hex_color


def create_idle_sim(mode, log_width, log_height):
    m = (mode or "").lower()
    if m == "rainbow":
        return IdleRainbow(log_width, log_height)
    elif m == "noise":
        return IdleNoise(log_width, log_height)
    elif m == "matrix":
        return IdleMatrixRain(log_width, log_height)
    elif m == "galaxy":
        return IdleGalaxy(log_width, log_height)
    elif m == "sparkle":
        return IdleSparkle(log_width, log_height)
    else:
        return None  # "off" or unknown


# ===== SIMULATION LOOP (PIXEL-ONLY) =====

def create_sim(cfg, log_W, log_H):
    mode = cfg["mode"]
    if mode == "pi":
        return MonteCarloPi(domain_size=cfg["pi_domain"], log_width=log_W, log_height=log_H)
    elif mode == "normal":
        return NormalHistogram(mu=cfg["mu"], sigma=cfg["sigma"], log_width=log_W, log_height=log_H)
    elif mode == "poisson":
        return PoissonHistogram(lam=cfg["lam"], log_width=log_W, log_height=log_H)
    elif mode == "random_walk":
        return RandomWalkSim(log_width=log_W, log_height=log_H)
    elif mode == "random_walk3":
        return RandomWalkSim3(log_width=log_W, log_height=log_H)
    elif mode == "row_race":
        return RowRaceSim(log_width=log_W, log_height=log_H)
    elif mode == "snake":
        return SnakeSim(log_width=log_W, log_height=log_H,
                        length=cfg.get("snake_length", 5), pause=cfg.get("snake_pause", 2.0))
    elif mode == "pacman":
        return PacManSim(log_width=log_W, log_height=log_H,
                         ghost_count=cfg.get("pacman_ghosts", 3))
    elif mode == "random_pixel":
        return RandomPixelSim(log_width=log_W, log_height=log_H)
    elif mode == "worm":
        return WormSim(log_width=log_W, log_height=log_H, length=cfg.get("worm_length", 16))
    elif mode == "column_race":
        return ColumnRaceSim(log_width=log_W, log_height=log_H)
    elif mode == "oscillator":
        return MultiStripOscillator(log_width=log_W, log_height=log_H)
    elif mode == "life":
        return GameOfLifeSim(log_width=log_W, log_height=log_H)
    elif mode == "heat":
        return HeatDiffusionSim(log_width=log_W, log_height=log_H)
    elif mode == "stock":
        return StockGBMSim(log_width=log_W, log_height=log_H)
    elif mode == "lorenz":
        return LorenzSim(log_width=log_W, log_height=log_H)
    else:
        return MonteCarloPi(domain_size=cfg["pi_domain"], log_width=log_W, log_height=log_H)


def simulation_loop():
    global status
    print("simulation_loop: starting background loop", flush=True)

    try:
        with config_lock:
            cfg = dict(config)

        log_W, log_H = get_logical_dims(cfg["rotation"])

        sim = create_sim(cfg, log_W, log_H)
        idle_sim = create_idle_sim(cfg.get("idle_mode", "rainbow"), log_W, log_H)

        last_mode   = cfg["mode"]
        last_domain = cfg["pi_domain"]
        last_mu     = cfg["mu"]
        last_sigma  = cfg["sigma"]
        last_lam    = cfg["lam"]
        last_rot    = cfg["rotation"]
        last_worm_length = cfg.get("worm_length", 16)
        last_snake_length = cfg.get("snake_length", 5)
        last_idle   = cfg.get("idle_mode", "rainbow")

        last_colors = ["000000"] * NLED
        changed_pixels_total = 0

        clear_matrix()

        while True:
            with config_lock:
                cfg = dict(config)

            log_W, log_H = get_logical_dims(cfg["rotation"])

            mode_changed   = (cfg["mode"] != last_mode)
            pi_changed     = (cfg["mode"] == "pi"      and cfg["pi_domain"] != last_domain)
            norm_changed   = (cfg["mode"] == "normal"  and (cfg["mu"] != last_mu or cfg["sigma"] != last_sigma))
            pois_changed   = (cfg["mode"] == "poisson" and cfg["lam"] != last_lam)
            rot_changed    = (cfg["rotation"] != last_rot)
            idle_changed   = (cfg.get("idle_mode", "rainbow") != last_idle)
            new_mode_extra = cfg["mode"] in ("heat", "stock", "lorenz") and mode_changed
            worm_length_changed = cfg["mode"] == "worm" and cfg.get("worm_length", 16) != last_worm_length
            snake_length_changed = cfg["mode"] == "snake" and cfg.get("snake_length", 5) != last_snake_length

            if (mode_changed or pi_changed or norm_changed or pois_changed or rot_changed or new_mode_extra
                    or worm_length_changed or snake_length_changed):
                sim = create_sim(cfg, log_W, log_H)

                last_mode   = cfg["mode"]
                last_domain = cfg["pi_domain"]
                last_mu     = cfg["mu"]
                last_sigma  = cfg["sigma"]
                last_lam    = cfg["lam"]
                last_rot    = cfg["rotation"]
                last_worm_length = cfg.get("worm_length", 16)
                last_snake_length = cfg.get("snake_length", 5)

                last_colors = ["000000"] * NLED
                changed_pixels_total = 0
                clear_matrix()

            if idle_changed or rot_changed:
                idle_sim = create_idle_sim(cfg.get("idle_mode", "rainbow"), log_W, log_H)
                last_idle = cfg.get("idle_mode", "rainbow")

            dt = 1.0 / max(1, cfg["fps"])
            frame_start = time.time()

            changed_this_frame = {}

            skip_frame_pixel_count = False
            if cfg["running"]:
                for _ in range(cfg["points_per_frame"]):
                    res = sim.sample_pixel_step()
                    if res is None:
                        continue
                    updates = res if isinstance(res, list) else [res]
                    skip_count = getattr(sim, "_last_patch_skip_count", False)
                    if skip_count:
                        skip_frame_pixel_count = True
                    for idx, hex_color in updates:
                        changed_this_frame[idx] = hex_color
                    if skip_count and hasattr(sim, "_last_patch_skip_count"):
                        sim._last_patch_skip_count = False
            else:
                if idle_sim is not None:
                    for _ in range(cfg["points_per_frame"]):
                        res = idle_sim.sample_pixel_step()
                        if res is None:
                            continue
                        idx, hex_color = res
                        changed_this_frame[idx] = hex_color

            if changed_this_frame:
                patch = []
                new_pixels = 0
                for idx, hex_color in changed_this_frame.items():
                    if last_colors[idx] == hex_color:
                        continue
                    last_colors[idx] = hex_color
                    patch.extend([idx, 1, hex_color])
                    new_pixels += 1
                    if patch:
                        # Debug: how many pixels are we sending?
                        # print(f"simulation_loop: sending {len(patch)//3} pixels", flush=True)
                        send_frame_batched(patch)
                        if cfg["running"] and not skip_frame_pixel_count and cfg.get("mode") not in ("snake", "pacman"):
                            changed_pixels_total += new_pixels

            if cfg["running"] and changed_pixels_total >= cfg["pixel_reset_after"]:
                with config_lock:
                    status["last_error"] = ""
                time.sleep(cfg["pixel_pause"])
                clear_matrix()
                last_colors = ["000000"] * NLED
                changed_pixels_total = 0
                sim.reset()
                if idle_sim is not None:
                    idle_sim.reset()

            with config_lock:
                if cfg["running"]:
                    status["stats"] = sim.stats_str()
                else:
                    status["stats"] = f"idle: {cfg.get('idle_mode', 'off')}"

            elapsed = time.time() - frame_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        # If this prints, the loop is crashing and exiting
        print("simulation_loop: CRASHED:", repr(e), file=sys.stderr, flush=True)
        with config_lock:
            status["last_error"] = f"simulation_loop crashed: {e!r}"



# ===== BACKGROUND SIMULATION THREAD STARTUP =====

_sim_thread_started = False
_sim_thread_lock = threading.Lock()

def start_simulation_thread():
    """
    Start the background simulation loop exactly once in this process.
    Safe to call multiple times; only the first call will spawn the thread.
    """
    global _sim_thread_started
    with _sim_thread_lock:
        if _sim_thread_started:
            return
        _sim_thread_started = True

        t = threading.Thread(target=simulation_loop, daemon=True)
        t.start()

# When running under a WSGI server (gunicorn), this module is imported
# and __name__ != '__main__'. We still want the simulation loop to run.
if __name__ != "__main__":
    start_simulation_thread()
    
# ===== FLASK APP =====

app = Flask(__name__)

def _start_background_sim():
    # This will run before every request, but the thread only starts once
    start_simulation_thread()
# Flask 3.x: use before_request (before_first_request removed)

@app.before_request
def _ensure_sim_thread():
    start_simulation_thread()

@app.route("/", methods=["GET", "POST"])
def index():
    global config, status

    if request.method == "POST":
        action = request.form.get("action")
        if action == "clear":
            clear_matrix()

        with config_lock:
            config["mode"] = request.form.get("mode", config["mode"])
            config["fps"] = int(request.form.get("fps", config["fps"]))
            config["points_per_frame"] = int(request.form.get("points_per_frame", config["points_per_frame"]))

            config["pi_domain"] = int(request.form.get("pi_domain", config["pi_domain"]))
            config["mu"] = float(request.form.get("mu", config["mu"]))
            config["sigma"] = float(request.form.get("sigma", config["sigma"]))
            config["lam"] = float(request.form.get("lam", config["lam"]))

            worm_length = int(request.form.get("worm_length", config.get("worm_length", 16)))
            worm_length = max(2, worm_length)
            worm_length = min(worm_length, PHYS_WIDTH * PHYS_HEIGHT)
            config["worm_length"] = worm_length
            snake_length = int(request.form.get("snake_length", config.get("snake_length", 5)))
            snake_length = max(3, snake_length)
            snake_length = min(snake_length, PHYS_WIDTH * PHYS_HEIGHT)
            config["snake_length"] = snake_length
            snake_pause = float(request.form.get("snake_pause", config.get("snake_pause", 2.0)))
            config["snake_pause"] = max(0.0, snake_pause)
            pacman_ghosts = int(request.form.get("pacman_ghosts", config.get("pacman_ghosts", 3)))
            pacman_ghosts = max(1, min(4, pacman_ghosts))
            config["pacman_ghosts"] = pacman_ghosts
            config["pixel_reset_after"] = int(request.form.get("pixel_reset_after", config["pixel_reset_after"]))
            config["pixel_pause"] = float(request.form.get("pixel_pause", config["pixel_pause"]))

            config["running"] = request.form.get("running", "true") == "true"

            config["palette"] = request.form.get("palette", config["palette"])
            config["idle_mode"] = request.form.get("idle_mode", config.get("idle_mode", "rainbow"))
            config["color_pattern"] = request.form.get("color_pattern", config.get("color_pattern", "data"))

            # single_color
            single = request.form.get("single_color", "#" + config.get("single_color", "ffffff"))
            single = single.strip()
            if single.startswith("#"):
                single = single[1:]
            single = single.lower()
            if len(single) != 6:
                single = config.get("single_color", "ffffff")
            config["single_color"] = single

            # Rotation
            rot_str = request.form.get("rotation", str(config["rotation"]))
            try:
                rot_val = int(rot_str)
            except ValueError:
                rot_val = config["rotation"]
            if rot_val not in (0, 90, 180, 270):
                rot_val = 0
            config["rotation"] = rot_val

    with config_lock:
        cfg_obj = type("Cfg", (), config.copy())
        stats = status["stats"]
        last_error = status["last_error"]

    return render_template("index.html", cfg=cfg_obj, stats=stats, last_error=last_error)


@app.route("/status")
def api_status():
    with config_lock:
        return jsonify({
            "config": config,
            "status": status,
        })


# ===== CLI / JSON CONFIG LOADING =====

def apply_external_config():
    parser = argparse.ArgumentParser(description="LED matrix Monte Carlo & art server")
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    parser.add_argument("--palette", help="Palette override",
                        choices=["fire", "plasma", "viridis", "turbo", "neon", "rainbow", "aurora", "random", "single"])
    parser.add_argument("--idle-mode", help="Idle mode override",
                        choices=["off", "rainbow", "noise", "matrix", "galaxy", "sparkle"])
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                data = json.load(f)
            with config_lock:
                for k, v in data.items():
                    if k in config:
                        config[k] = v
        except Exception as e:
            print(f"Error loading config JSON: {e}")

    with config_lock:
        if args.palette:
            config["palette"] = args.palette
        if args.idle_mode:
            config["idle_mode"] = args.idle_mode



if __name__ == "__main__":
    apply_external_config()

    # Start simulation loop in dev mode
    start_simulation_thread()

    print("Starting pixel-only LED sim UI on http://localhost:5030")
    print("Physical panel: 64x16; logical grid switches between 64x16 and 16x64 with rotation.")
    app.run(host="0.0.0.0", port=5030, debug=False)

