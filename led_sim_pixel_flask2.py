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
    - life      : Conway's Game of Life
    - heat      : Heat diffusion / reaction-diffusion-like pattern
    - stock     : Stock price Monte Carlo (GBM sparkline)
    - lorenz    : Lorenz attractor projected onto grid

Other features:
    - Pixel build: one-pixel-at-a-time, sending only changed pixels
    - Auto-reset after N changed pixels (with pause and clear)
    - Color palettes: fire, plasma, viridis, turbo, neon, single
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
    # "heat", "stock", "lorenz"
    "mode": "pi",
    "fps": 15,
    "points_per_frame": 20,

    "pi_domain": 16,           # pixels, square for Monte Carlo π
    "mu": 0.0,                 # normal mean
    "sigma": 1.0,              # normal std dev
    "lam": 5.0,                # poisson lambda

    "pixel_reset_after": 800,  # changed pixels before reset (for main sims)
    "pixel_pause": 2.0,        # seconds to pause before clear+restart

    # Rotation angle in degrees: 0, 90, 180, or 270
    "rotation": 0,

    # Color palette / theme
    # "fire", "plasma", "viridis", "turbo", "neon", "single"
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

        if mode_changed or pi_changed or norm_changed or pois_changed or rot_changed or new_mode_extra:
            sim = create_sim(cfg, log_W, log_H)

            last_mode   = cfg["mode"]
            last_domain = cfg["pi_domain"]
            last_mu     = cfg["mu"]
            last_sigma  = cfg["sigma"]
            last_lam    = cfg["lam"]
            last_rot    = cfg["rotation"]

            last_colors = ["000000"] * NLED
            changed_pixels_total = 0
            clear_matrix()

        if idle_changed or rot_changed:
            idle_sim = create_idle_sim(cfg.get("idle_mode", "rainbow"), log_W, log_H)
            last_idle = cfg.get("idle_mode", "rainbow")

        dt = 1.0 / max(1, cfg["fps"])
        start = time.time()

        changed_this_frame = {}

        if cfg["running"]:
            for _ in range(cfg["points_per_frame"]):
                res = sim.sample_pixel_step()
                if res is None:
                    continue
                idx, hex_color = res
                changed_this_frame[idx] = hex_color
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
            for idx, hex_color in changed_this_frame.items():
                if last_colors[idx] == hex_color:
                    continue
                last_colors[idx] = hex_color
                patch.extend([idx, 1, hex_color])
                if cfg["running"]:
                    changed_pixels_total += 1
            if patch:
                send_frame_batched(patch)

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

        elapsed = time.time() - start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

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

# When running under a WSGI server (gunicorn, uWSGI, etc.), this will be
# called automatically on the first HTTP request.

app = Flask(__name__)



# ===== FLASK APP =====




@app.before_request
def _start_background_sim():
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
                        choices=["fire", "plasma", "viridis", "turbo", "neon", "single"])
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
    # Only parse CLI args when running as a script (dev mode)
    apply_external_config()

    # Start simulation loop in background
    start_simulation_thread()

    print("Starting pixel-only LED sim UI on http://localhost:5000")
    print("Physical panel: 64x16; logical grid switches between 64x16 and 16x64 with rotation.")


    print("Starting pixel-only LED sim UI on http://localhost:5000")
    print("Physical panel: 64x16; logical grid switches between 64x16 and 16x64 with rotation.")
    app.run(host="0.0.0.0", port=5030, debug=False)
