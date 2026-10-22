#!/usr/bin/env python3
"""
A rotating, pulsating 3D heart made of the words "i love you".
Single color: #c30f16 (needs a truecolor terminal). Press Ctrl+C to stop.
"""

import math
import os
import shutil
import sys
import time

TEXT = "i love you "
COLOR = (0xC3, 0x0F, 0x16)  # #c30f16
FPS = 60
BEAT_PERIOD = 0.6       # seconds per heartbeat
ROTATION_SPEED = 0.5    # radians per second
TILT = 0.50             # slight forward tilt so you see the top of the lobes

# ---------------------------------------------------------------- the heart
# Implicit 3D heart surface (y is up, z is depth):
#   (x^2 + 9/4 z^2 + y^2 - 1)^3 - x^2 y^3 - 9/80 z^2 y^3 <= 0  -> inside
def inside(x, y, z):
    a = x * x + 2.25 * z * z + y * y - 1.0
    return a * a * a - x * x * y ** 3 - 0.1125 * z * z * y ** 3 <= 0


def build_surface(nx=64, ny=64, nz=40):
    """Voxelize the heart; return surface points as (x, y, z, nx, ny, nz, char)."""
    xr, yr, zr = 1.3, 1.3, 0.8
    xs = [-xr + 2 * xr * i / (nx - 1) for i in range(nx)]
    ys = [-yr + 2 * yr * j / (ny - 1) for j in range(ny)]
    zs = [-zr + 2 * zr * k / (nz - 1) for k in range(nz)]

    vox = [[[inside(xs[i], ys[j], zs[k]) for k in range(nz)]
            for j in range(ny)] for i in range(nx)]

    def solid(i, j, k):
        return 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and vox[i][j][k]

    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    rows = {}  # j -> list of points, so letters can run around the heart in rings
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not vox[i][j][k]:
                    continue
                nxv = nyv = nzv = 0.0
                for dx, dy, dz in dirs:
                    if not solid(i + dx, j + dy, k + dz):
                        nxv += dx
                        nyv += dy
                        nzv += dz
                length = math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv)
                if length == 0:
                    continue  # interior voxel
                rows.setdefault(j, []).append(
                    (xs[i], ys[j], zs[k], nxv / length, nyv / length, nzv / length))

    # Assign letters: in every horizontal ring, order points around the
    # vertical axis so the words read around the heart as it spins.
    points = []
    counter = 0
    for j in sorted(rows):
        ring = sorted(rows[j], key=lambda p: math.atan2(p[2], p[0]))
        for p in ring:
            points.append(p + (TEXT[counter % len(TEXT)],))
            counter += 1
        counter += 3  # offset between rings so words don't line up in columns
    return points


# ------------------------------------------------------------------ pulsing
def pulse_scale(t):
    """Double-beat 'lub-dub', scale between ~0.82 and 1.0."""
    phase = (t % BEAT_PERIOD) / BEAT_PERIOD
    b1 = math.exp(-((phase - 0.10) ** 2) / 0.0035)
    b2 = math.exp(-((phase - 0.30) ** 2) / 0.0050) * 0.7
    return 0.82 + 0.18 * min(1.0, b1 + b2)


# ---------------------------------------------------------------- rendering
LIGHT = (-0.35, 0.55, 0.76)
_l = math.sqrt(sum(c * c for c in LIGHT))
LIGHT = tuple(c / _l for c in LIGHT)


def render(points, width, height, t):
    scale = pulse_scale(t)
    ang = t * ROTATION_SPEED
    ca, sa = math.cos(ang), math.sin(ang)
    ct, st = math.cos(TILT), math.sin(TILT)

    cx, cy = width / 2, height / 2
    unit_y = (height / 2.9) * scale
    unit_x = unit_y * 2.0        # terminal cells are ~twice as tall as wide
    cam = 4.0                    # camera distance for mild perspective

    zbuf = {}
    for x, y, z, nx, ny, nz, ch in points:
        # rotate around the vertical axis, then tilt around the horizontal axis
        x1 = x * ca + z * sa
        z1 = -x * sa + z * ca
        y2 = y * ct - z1 * st
        z2 = y * st + z1 * ct

        nx1 = nx * ca + nz * sa
        nz1 = -nx * sa + nz * ca
        ny2 = ny * ct - nz1 * st
        nz2 = ny * st + nz1 * ct

        persp = cam / (cam - z2)
        col = int(cx + x1 * persp * unit_x)
        row = int(cy - (y2 - 0.1) * persp * unit_y)
        if not (0 <= col < width and 0 <= row < height):
            continue

        key = (row, col)
        if key in zbuf and zbuf[key][0] >= z2:
            continue

        # Lambert shading from the surface normal -> 3 brightness tiers
        shade = max(0.0, nx1 * LIGHT[0] + ny2 * LIGHT[1] + nz2 * LIGHT[2])
        tier = 2 if shade > 0.55 else (1 if shade > 0.2 else 0)
        zbuf[key] = (z2, ch, tier)

    # Compose frame. Color is always #c30f16; depth/shading comes from
    # bold (lit), normal, and faint (shadow) text weight.
    weight = {2: "\033[1m", 1: "\033[22m", 0: "\033[2m"}
    out = []
    for row in range(height):
        line = []
        cur = None
        for col in range(width):
            cell = zbuf.get((row, col))
            if cell is None:
                line.append(" ")
                continue
            _, ch, tier = cell
            if tier != cur:
                line.append(weight[tier])
                cur = tier
            line.append(ch)
        line.append("\033[22m")
        out.append("".join(line).rstrip())
    return "\n".join(out)


def main():
    if os.name == "nt":
        os.system("")  # enable ANSI escape codes on Windows terminals

    print("Building heart...", end="", flush=True)
    points = build_surface()

    HIDE, SHOW = "\033[?25l", "\033[?25h"
    HOME, CLEAR = "\033[H", "\033[2J"
    RESET = "\033[0m"
    color = "\033[38;2;{};{};{}m".format(*COLOR)

    sys.stdout.write(CLEAR + HIDE)
    start = time.time()
    try:
        while True:
            frame_start = time.time()
            size = shutil.get_terminal_size((80, 24))
            width, height = size.columns, max(size.lines - 2, 10)

            frame = render(points, width, height, frame_start - start)
            sys.stdout.write(HOME + color + frame + RESET + "\033[J")
            sys.stdout.flush()

            delay = 1 / FPS - (time.time() - frame_start)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET + SHOW + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
