#!/usr/bin/env python3
"""
Interactive 3D anatomical heart made of the words "i love you" (pygame).

  python -m pip install pygame
  python heart_pygame_fixed.py

Controls: drag = orbit | wheel = zoom | space = pause spin | S = save PNG | Esc = quit
If anything goes wrong the error is printed and the window waits for Enter,
so the message can't vanish.
"""

import math
import sys
import time

TEXT = "i love you "
COLOR = (0xC3, 0x0F, 0x16)   # #c30f16
FPS = 60
BEAT_PERIOD = 1.2            # seconds per heartbeat
ROTATION_SPEED = 0.45        # radians per second (auto-spin; 0 = fixed front view)
TILT = 0.12                  # slight forward tilt
FONT_NAME = None             # None = pygame's built-in font (always works, no slow font scan)
                             # or a system font name such as "consolas" / "couriernew"
STEP = 0.03                  # voxel size of the model (smaller = denser text, slower start)

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
        ring = sorted(rows[j], key=lambda p: -math.atan2(p[2] + 0.1, p[0]))
        for p in ring:
            # centre the model on its rotation axis
            points.append((p[0], p[1] - 0.15, p[2] + 0.1, p[3], p[4], p[5],
                           TEXT[counter % len(TEXT)]))
            counter += 1
        counter += 3
    return points


# ------------------------------------------------------------------ pulsing
def pulse_scale(t):
    """Double-beat 'lub-dub', scale between ~0.88 and 1.0."""
    phase = (t % BEAT_PERIOD) / BEAT_PERIOD
    b1 = math.exp(-((phase - 0.10) ** 2) / 0.0035)
    b2 = math.exp(-((phase - 0.30) ** 2) / 0.0050) * 0.7
    return 0.88 + 0.12 * min(1.0, b1 + b2)


# ---------------------------------------------------------- 3D projection
LIGHT = norm3((-0.45, 0.60, 0.66))   # key light: upper left, toward the viewer
AMBIENT = 0.40                       # darkest shadow = 40% of the material color
SHADE_LEVELS = 14                    # brightness buckets (glyph cache size)
GLYPH_SCALE = 1.55                   # glyph size relative to point spacing
CAMERA = 4.5                         # camera distance (perspective strength)
BG = (11, 10, 14)


def project(points, w, h, yaw, pitch, beat, zoom):
    """Rotate, light and perspective-project every surface point.

    Returns a back-to-front draw list of (x, y, char, font_px, shade_level).
    The material color is never brightened beyond #c30f16 -- lighting can
    only darken it.
    """
    ca, sa = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    lx, ly, lz = LIGHT
    unit = min(w, h) / 3.0 * zoom * beat          # pixels per model unit
    spacing = STEP * unit
    cx, cy = w / 2, h / 2
    top = SHADE_LEVELS - 1

    out = []
    for x, y, z, nx, ny, nz, ch in points:
        # normal first: skip faces turned away from the camera
        nz1 = -nx * sa + nz * ca
        nz2 = ny * sp + nz1 * cp
        if nz2 < -0.12:
            continue
        nx1 = nx * ca + nz * sa
        ny2 = ny * cp - nz1 * sp

        x1 = x * ca + z * sa
        z1 = -x * sa + z * ca
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        persp = CAMERA / (CAMERA - z2)
        sx = cx + x1 * persp * unit
        sy = cy - y2 * persp * unit
        if not (-20 < sx < w + 20 and -20 < sy < h + 20):
            continue

        lam = (nx1 * lx + ny2 * ly + nz2 * lz) * 1.6
        if lam < 0.0:
            lam = 0.0
        elif lam > 1.0:
            lam = 1.0
        depth = min(1.0, max(0.0, (z2 + 1.1) / 2.2))      # nearer = a bit brighter
        s = (AMBIENT + (1.0 - AMBIENT) * lam) * (0.82 + 0.18 * depth)
        level = int(min(1.0, s) * top + 0.5)

        px = max(6, int(spacing * persp * GLYPH_SCALE))
        out.append((z2, sx, sy, ch, px, level))

    out.sort()                                             # painter's algorithm
    return [(sx, sy, ch, px, lv) for _, sx, sy, ch, px, lv in out]


def level_color(level):
    k = (level / (SHADE_LEVELS - 1))
    return (int(COLOR[0] * k), int(COLOR[1] * k), int(COLOR[2] * k))


# --------------------------------------------------------------- the window
def pause(msg=""):
    if msg:
        print(msg)
    try:
        input("\nPress Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def run():
    print("Python", sys.version.split()[0], "-", sys.executable, flush=True)
    try:
        import pygame
    except ImportError:
        pause("pygame is NOT installed for this Python. Install it with:\n"
              '  "%s" -m pip install pygame' % sys.executable)
        return
    print("pygame", pygame.version.ver, flush=True)

    pygame.init()
    pygame.display.set_caption("i love you - anatomical heart")
    screen = pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    screen.fill(BG)
    pygame.display.flip()
    pygame.event.pump()
    print("Window opened. Building heart model (about a second)...", flush=True)
    points = extract_surface(build_model())
    print("Running. Close the window or press Esc to quit.", flush=True)

    clock = pygame.time.Clock()
    wheel_event = getattr(pygame, "MOUSEWHEEL", None)   # missing in very old pygame

    fonts = {}
    glyphs = {}

    def glyph(ch, px, level):
        key = (ch, px, level)
        g = glyphs.get(key)
        if g is None:
            f = fonts.get(px)
            if f is None:
                if FONT_NAME is None:
                    f = pygame.font.Font(None, px)
                else:
                    f = pygame.font.SysFont(FONT_NAME, px, bold=True)
                fonts[px] = f
            g = glyphs[key] = f.render(ch, True, level_color(level))
        return g

    yaw, pitch, zoom = 0.0, TILT, 1.0
    spinning, dragging = ROTATION_SPEED != 0, False
    t0 = time.time()
    last = t0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                spinning = not spinning
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(screen, "heart.png")
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                yaw += e.rel[0] * 0.01
                pitch = max(-1.4, min(1.4, pitch + e.rel[1] * 0.01))
            elif wheel_event is not None and e.type == wheel_event:
                zoom = max(0.4, min(3.0, zoom * (1.1 ** e.y)))

        now = time.time()
        if spinning and not dragging:
            yaw += ROTATION_SPEED * (now - last)
        last = now

        w, h = screen.get_size()
        draw = project(points, w, h, yaw, pitch, pulse_scale(now - t0), zoom)

        screen.fill(BG)
        blit = screen.blit
        for sx, sy, ch, px, lv in draw:
            g = glyph(ch, px, lv)
            blit(g, (int(sx - g.get_width() * 0.5), int(sy - g.get_height() * 0.5)))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def main():
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        pause("\nSomething went wrong -- the error is shown above.")


if __name__ == "__main__":
    main()
