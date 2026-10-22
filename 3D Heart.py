#!/usr/bin/env python3
"""
Anatomically modeled, fully 3D rendered human heart made of the words
"i love you", shown live in a plain Python (tkinter) window.

  pip install pillow        (tkinter already ships with Python)
  python pulsating_heart_window.py

No controls and no files: just the heart, turning and beating.
Close the window to quit.

Material color is #c30f16: the brightest lit surface is exactly that color,
shadowed areas are the same color, darker -- no other hues are used.
"""

import math
import sys
import time

TEXT = "i love you "
COLOR = (0xC3, 0x0F, 0x16)   # #c30f16
BEAT_PERIOD = 1.2            # seconds per heartbeat
TILT = 0.12                  # slight forward tilt
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
SHADE_LEVELS = 14                    # brightness steps
GLYPH_SCALE = 1.55                   # glyph size relative to point spacing
CAMERA = 4.5                         # camera distance (perspective strength)
BG = (11, 10, 14)

# ---------------------------------------------------------- window setup
WINDOW_SIZE = 680        # pixels (square); smaller = smoother on slow machines
SUPERSAMPLE = 1          # 2 = smoother text but roughly half the frame rate
ROTATION_SPEED = 0.5     # radians per second


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
    k = level / (SHADE_LEVELS - 1)
    return (int(COLOR[0] * k), int(COLOR[1] * k), int(COLOR[2] * k))


# --------------------------------------------------------------- rendering
FONT_CANDIDATES = [
    "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "consolab.ttf",
    "Consolas Bold.ttf", "courbd.ttf", "Courier New Bold.ttf", "Menlo.ttc",
    "FreeMonoBold.ttf", "DejaVuSansMono.ttf", "cour.ttf",
]


class GlyphCache:
    """Anti-aliased glyph masks, made once per (character, size)."""

    def __init__(self):
        self.fonts = {}
        self.masks = {}
        self.font_name = None

    def font(self, px):
        f = self.fonts.get(px)
        if f is not None:
            return f
        from PIL import ImageFont
        names = [self.font_name] if self.font_name else FONT_CANDIDATES
        for name in names:
            try:
                f = ImageFont.truetype(name, px)
                self.font_name = name
                break
            except OSError:
                continue
        if f is None:                       # last resort: Pillow's built-in font
            try:
                f = ImageFont.load_default(px)
            except TypeError:
                f = ImageFont.load_default()
        self.fonts[px] = f
        return f

    def mask(self, ch, px):
        key = (ch, px)
        m = self.masks.get(key)
        if m is None:
            from PIL import Image, ImageDraw
            m = Image.new("L", (px, px), 0)
            ImageDraw.Draw(m).text((px / 2, px / 2), ch, font=self.font(px),
                                   fill=255, anchor="mm")
            self.masks[key] = m
        return m


def render_frame(points, size, yaw, pitch, beat, cache):
    from PIL import Image
    big = size * SUPERSAMPLE
    img = Image.new("RGB", (big, big), BG)
    paste = img.paste
    for sx, sy, ch, px, lv in project(points, big, big, yaw, pitch, beat, 1.0):
        paste(level_color(lv), (int(sx - px / 2), int(sy - px / 2)), cache.mask(ch, px))
    if SUPERSAMPLE > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img



def main():
    try:
        import PIL  # noqa: F401
    except ImportError:
        sys.exit("This needs Pillow:  pip install pillow")
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("This needs tkinter (it ships with Python; on Linux install python3-tk).")

    points = extract_surface(build_model())
    cache = GlyphCache()

    root = tk.Tk()
    root.title("i love you")
    root.resizable(False, False)
    bg_hex = "#%02x%02x%02x" % BG
    root.configure(bg=bg_hex)
    label = tk.Label(root, bd=0, highlightthickness=0, bg=bg_hex)
    label.pack()

    start = time.time()

    def tick():
        t = time.time() - start
        frame = render_frame(points, WINDOW_SIZE, t * ROTATION_SPEED, TILT,
                             pulse_scale(t), cache)
        # Hand the frame to Tk as a PPM image (no extra imaging libs needed)
        header = b"P6 %d %d 255\n" % frame.size
        photo = tk.PhotoImage(data=header + frame.tobytes(), format="PPM")
        label.configure(image=photo)
        label.image = photo            # keep a reference so it isn't garbage collected
        root.after(1, tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
