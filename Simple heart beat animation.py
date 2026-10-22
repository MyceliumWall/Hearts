
import math
import os
import shutil
import sys
import time

TEXT = "i love you "
COLOR = (0xC3, 0x0F, 0x16)   # #c30f16
FPS = 20
BEAT_PERIOD = 1.2            # seconds per heartbeat
ROTATION_SPEED = 0.5         # radians per second (set 0 for a fixed front view)
TILT = 0.12                  # slight forward tilt
STEP = 0.03                  # voxel size of the model (smaller = finer, slower start)

# Model-space bounds (x: patient's right -> left, y: up, z: back -> front)
X0, X1 = -1.20, 1.35
Y0, Y1 = -1.30, 1.60
Z0, Z1 = -1.15, 0.90
NX = int((X1 - X0) / STEP) + 1
NY = int((Y1 - Y0) / STEP) + 1
NZ = int((Z1 - Z0) / STEP) + 1


# ----------------------------------------------------------- voxel modeling
class Grid:
    def __init__(self):
        self.v = bytearray(NX * NY * NZ)
        self.xs = [X0 + i * STEP for i in range(NX)]
        self.ys = [Y0 + j * STEP for j in range(NY)]
        self.zs = [Z0 + k * STEP for k in range(NZ)]

    def rng(self, lo, hi, org, n):
        a = max(0, int(math.floor((lo - org) / STEP)))
        b = min(n - 1, int(math.ceil((hi - org) / STEP)))
        return range(a, b + 1)

    def mark(self, i, j, k):
        self.v[(i * NY + j) * NZ + k] = 1

    def solid(self, i, j, k):
        return self.v[(i * NY + j) * NZ + k]


def norm3(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n)


def add_spheroid(g, c, axis, a, b):
    """Prolate spheroid: semi-length a along `axis`, semi-width b across it."""
    ux, uy, uz = norm3(axis)
    r = max(a, b)
    ia = 1.0 / (a * a)
    ib = 1.0 / (b * b)
    for i in g.rng(c[0] - r, c[0] + r, X0, NX):
        dx = g.xs[i] - c[0]
        for j in g.rng(c[1] - r, c[1] + r, Y0, NY):
            dy = g.ys[j] - c[1]
            for k in g.rng(c[2] - r, c[2] + r, Z0, NZ):
                dz = g.zs[k] - c[2]
                p = dx * ux + dy * uy + dz * uz
                q = dx * dx + dy * dy + dz * dz - p * p
                if p * p * ia + q * ib <= 1.0:
                    g.mark(i, j, k)


def add_ellipsoid(g, c, r3):
    rx, ry, rz = r3
    for i in g.rng(c[0] - rx, c[0] + rx, X0, NX):
        dx = (g.xs[i] - c[0]) / rx
        for j in g.rng(c[1] - ry, c[1] + ry, Y0, NY):
            dy = (g.ys[j] - c[1]) / ry
            for k in g.rng(c[2] - rz, c[2] + rz, Z0, NZ):
                dz = (g.zs[k] - c[2]) / rz
                if dx * dx + dy * dy + dz * dz <= 1.0:
                    g.mark(i, j, k)


def add_sphere(g, c, r):
    add_ellipsoid(g, c, (r, r, r))


def catmull_rom(pts, per_seg):
    """Smooth curve through control points; returns (points, segment, t)."""
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for s in range(len(pts) - 1):
        p0, p1, p2, p3 = p[s], p[s + 1], p[s + 2], p[s + 3]
        for n in range(per_seg + (1 if s == len(pts) - 2 else 0)):
            t = n / per_seg
            t2, t3 = t * t, t * t * t
            pt = tuple(
                0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t
                       + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2
                       + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)
                for d in range(3))
            out.append((pt, s, t))
    return out


def add_tube(g, pts, radii):
    """Tube of varying radius following a smooth curve through `pts`."""
    # rough length -> enough samples that stamped spheres overlap
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    per_seg = max(4, int(length / (len(pts) - 1) / (STEP * 0.5)))
    for pt, s, t in catmull_rom(pts, per_seg):
        r = radii[s] + (radii[s + 1] - radii[s]) * t
        add_sphere(g, pt, r)


def front_surface_z(g, x, y):
    """z of the front-most solid voxel at (x, y), or None."""
    i = int(round((x - X0) / STEP))
    j = int(round((y - Y0) / STEP))
    if not (0 <= i < NX and 0 <= j < NY):
        return None
    for k in range(NZ - 1, -1, -1):
        if g.solid(i, j, k):
            return g.zs[k]
    return None


def add_surface_vessel(g, xy_pts, r):
    """Coronary artery: runs along the front surface of the heart."""
    pts3 = [(x, y, 0.0) for x, y in xy_pts]
    for pt, _, _ in catmull_rom(pts3, 10):
        z = front_surface_z(g, pt[0], pt[1])
        if z is not None:
            add_sphere(g, (pt[0], pt[1], z - 0.01), r)


def build_model():
    g = Grid()
    axis = (0.50, -0.85, 0.18)  # long axis of the ventricles: apex points down, left, forward

    # --- ventricles --------------------------------------------------------
    add_spheroid(g, (0.26, -0.12, -0.05), axis, 1.00, 0.50)   # left ventricle (apex)
    add_spheroid(g, (-0.10, -0.15, 0.28), axis, 0.72, 0.40)   # right ventricle (front)

    # --- atria ---------------------------------------------------------------
    add_tube(g, [(-0.62, 0.38, 0.02), (-0.61, 0.08, 0.04), (-0.60, -0.20, 0.05)],
             [0.27, 0.31, 0.27])                                # right atrium
    add_ellipsoid(g, (-0.34, 0.40, 0.33), (0.20, 0.15, 0.14))   # right auricle
    add_ellipsoid(g, (0.20, 0.30, -0.50), (0.58, 0.27, 0.28))   # left atrium (back)
    add_tube(g, [(0.58, 0.30, -0.40), (0.60, 0.32, -0.20), (0.60, 0.33, -0.05)],
             [0.13, 0.14, 0.15])                                # left atrial appendage

    # --- aorta: root, ascending, arch, descending -------------------------
    add_sphere(g, (0.02, 0.22, 0.02), 0.21)                     # aortic root
    add_tube(g, [(0.02, 0.18, 0.02), (-0.01, 0.55, 0.0), (0.0, 0.88, -0.08),
                 (0.22, 1.02, -0.28), (0.46, 0.95, -0.52), (0.55, 0.70, -0.70),
                 (0.55, 0.30, -0.78)],
             [0.20, 0.15, 0.15, 0.14, 0.13, 0.125, 0.12])
    # arch branches: brachiocephalic, left common carotid, left subclavian
    add_tube(g, [(-0.01, 0.94, -0.07), (-0.08, 1.12, -0.06), (-0.10, 1.32, -0.04)],
             [0.07, 0.06, 0.06])
    add_tube(g, [(0.10, 1.01, -0.18), (0.14, 1.18, -0.16), (0.15, 1.36, -0.14)],
             [0.055, 0.05, 0.05])
    add_tube(g, [(0.30, 1.02, -0.34), (0.40, 1.14, -0.30), (0.52, 1.30, -0.28)],
             [0.06, 0.055, 0.055])

    # --- pulmonary trunk and arteries ---------------------------------------
    add_tube(g, [(0.08, 0.05, 0.34), (0.20, 0.38, 0.30), (0.26, 0.60, 0.12),
                 (0.30, 0.72, -0.04)],
             [0.15, 0.14, 0.13, 0.12])
    add_tube(g, [(0.30, 0.72, -0.04), (0.50, 0.76, -0.12), (0.74, 0.74, -0.22)],
             [0.12, 0.10, 0.09])                                # left pulmonary artery
    add_tube(g, [(0.30, 0.72, -0.04), (0.08, 0.72, -0.30), (-0.30, 0.74, -0.30),
                 (-0.62, 0.72, -0.28)],
             [0.12, 0.10, 0.09, 0.09])                          # right pulmonary artery

    # --- venae cavae -----------------------------------------------------------
    add_tube(g, [(-0.60, 0.30, 0.02), (-0.58, 0.80, 0.0), (-0.56, 1.25, -0.02)],
             [0.12, 0.115, 0.11])                               # superior vena cava
    add_tube(g, [(-0.60, -0.22, 0.04), (-0.55, -0.50, -0.02), (-0.50, -0.70, -0.08)],
             [0.13, 0.13, 0.13])                                # inferior vena cava

    # --- pulmonary veins (4) -------------------------------------------------
    add_tube(g, [(-0.10, 0.38, -0.52), (-0.45, 0.45, -0.56), (-0.80, 0.52, -0.56)],
             [0.08, 0.08, 0.08])
    add_tube(g, [(-0.10, 0.18, -0.56), (-0.45, 0.14, -0.58), (-0.80, 0.08, -0.58)],
             [0.08, 0.08, 0.08])
    add_tube(g, [(0.60, 0.38, -0.48), (0.90, 0.50, -0.50), (1.10, 0.62, -0.50)],
             [0.08, 0.08, 0.08])
    add_tube(g, [(0.60, 0.20, -0.54), (0.90, 0.14, -0.56), (1.10, 0.10, -0.56)],
             [0.08, 0.08, 0.08])

    # --- coronary arteries (on the heart's front surface) --------------------
    add_surface_vessel(g, [(0.18, 0.15), (0.26, -0.20), (0.38, -0.55), (0.52, -0.88)],
                       0.032)                                   # LAD
    add_surface_vessel(g, [(0.28, 0.22), (0.50, 0.14), (0.68, -0.05)],
                       0.030)                                   # circumflex
    add_surface_vessel(g, [(-0.16, 0.30), (-0.38, 0.28), (-0.50, 0.05),
                           (-0.50, -0.25), (-0.40, -0.58), (-0.20, -0.85)],
                       0.032)                                   # RCA
    return g


# ------------------------------------------------- surface points + normals
def extract_surface(g):
    """Surface voxels with smooth normals and letters assigned in rings."""
    R = 3
    offs = [(a, b, c) for a in range(-R, R + 1) for b in range(-R, R + 1)
            for c in range(-R, R + 1) if (a, b, c) != (0, 0, 0)
            and a * a + b * b + c * c <= R * R + 1]
    v = g.v
    sy = NZ
    sx = NY * NZ
    rows = {}
    for i in range(R, NX - R):
        for j in range(R, NY - R):
            base = i * sx + j * sy
            for k in range(R, NZ - R):
                if not v[base + k]:
                    continue
                if (v[base + k + 1] and v[base + k - 1] and v[base + sy + k]
                        and v[base - sy + k] and v[base + sx + k] and v[base - sx + k]):
                    continue  # interior voxel
                nx = ny = nz = 0.0
                for a, b, c in offs:
                    if v[base + a * sx + b * sy + c + k]:
                        nx -= a
                        ny -= b
                        nz -= c
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                if ln < 1e-9:
                    nx, ny, nz, ln = 0.0, 0.0, 1.0, 1.0
                rows.setdefault(j, []).append(
                    (g.xs[i], g.ys[j], g.zs[k], nx / ln, ny / ln, nz / ln))

    # Letters run around the heart in horizontal rings, so the words read
    # along the surface and turn with it.
    points = []
    counter = 0
    for j in sorted(rows):
        ring = sorted(rows[j], key=lambda p: math.atan2(p[2] + 0.1, p[0]))
        for p in ring:
            # centre the model on its rotation axis
            points.append((p[0], p[1] - 0.15, p[2] + 0.1, p[3], p[4], p[5],
                           TEXT[counter % len(TEXT)]))
            counter += 1
        counter += 3
    return points


# ------------------------------------------------------------------ pulsing
def pulse_scale(t):
    """Double-beat 'lub-dub', scale between ~0.86 and 1.0."""
    phase = (t % BEAT_PERIOD) / BEAT_PERIOD
    b1 = math.exp(-((phase - 0.10) ** 2) / 0.0035)
    b2 = math.exp(-((phase - 0.30) ** 2) / 0.0050) * 0.7
    return 0.86 + 0.14 * min(1.0, b1 + b2)


# ---------------------------------------------------------------- rendering
LIGHT = norm3((-0.35, 0.55, 0.76))


def render(points, width, height, t, angle=None):
    scale = pulse_scale(t)
    ang = t * ROTATION_SPEED if angle is None else angle
    ca, sa = math.cos(ang), math.sin(ang)
    ct, st = math.cos(TILT), math.sin(TILT)

    cx, cy = width / 2, height / 2
    unit_y = (height / 3.0) * scale
    unit_x = unit_y * 2.0         # terminal cells are ~twice as tall as wide
    cam = 4.5                     # camera distance for mild perspective
    lx, ly, lz = LIGHT

    zbuf = {}
    for x, y, z, nx, ny, nz, ch in points:
        x1 = x * ca + z * sa
        z1 = -x * sa + z * ca
        y2 = y * ct - z1 * st
        z2 = y * st + z1 * ct

        persp = cam / (cam - z2)
        col = int(cx + x1 * persp * unit_x)
        row = int(cy - y2 * persp * unit_y)
        if not (0 <= col < width and 0 <= row < height):
            continue

        key = row * width + col
        old = zbuf.get(key)
        if old is not None and old[0] >= z2:
            continue

        nx1 = nx * ca + nz * sa
        nz1 = -nx * sa + nz * ca
        ny2 = ny * ct - nz1 * st
        nz2 = ny * st + nz1 * ct
        shade = max(0.0, nx1 * lx + ny2 * ly + nz2 * lz)
        tier = 2 if shade > 0.55 else (1 if shade > 0.2 else 0)
        zbuf[key] = (z2, ch, tier)

    # Color is always #c30f16; shading comes from text weight:
    # bold (lit), normal, faint (shadow).
    weight = {2: "\033[1m", 1: "\033[22m", 0: "\033[2m"}
    out = []
    for row in range(height):
        line = []
        cur = None
        base = row * width
        for col in range(width):
            cell = zbuf.get(base + col)
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

    print("Building heart...", flush=True)
    points = extract_surface(build_model())

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
