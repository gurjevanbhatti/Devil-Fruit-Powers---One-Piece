"""
The rubber maths for Gum Gum: bending the real picture of your skin.

- ARAP, as-rigid-as-possible shape manipulation (Igarashi, Moscovich & Hughes,
  SIGGRAPH 2005). Bends a triangle mesh so every triangle stays as close to its
  rest shape as it can while a few handle vertices get dragged around.
- meshes: a strip along a finger, a forearm or a bit of cheek
- cutting a finger out of the picture by its skin colour
- drawing a bent mesh: each triangle of the real picture is warped into place
- filling in where the real finger / cheek was, so you don't see it twice

Everything is in camera pixels.
"""

import math

import cv2
import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------- ARAP ----------


class ARAP:
    """
    Two steps, both set up once per grab so each frame is cheap:
      1. let each triangle rotate and scale freely to fit the handles
      2. keep just the rotation from step 1 and fit the rest-length edges to it
    """

    def __init__(self, verts, tris, handles, weight=1000.0):
        verts = np.asarray(verts, np.float64)
        self.tris = np.asarray(tris, np.int64)
        self.n = n = len(verts)
        self.w = weight

        # step 1: how far each triangle is from a similarity transform of its rest shape
        A1 = np.zeros((2 * n, 2 * n))
        ginv = []
        dofs = []
        for t in self.tris:
            G = np.zeros((6, 4))
            for r, (px, py) in enumerate(verts[t]):
                G[2 * r] = (px, -py, 1, 0)
                G[2 * r + 1] = (py, px, 0, 1)
            gi = np.linalg.solve(G.T @ G, G.T)  # pulls (a, b, tx, ty) out of the 6 coords
            F = np.eye(6) - G @ gi
            d = np.array([k for v in t for k in (2 * v, 2 * v + 1)])
            A1[np.ix_(d, d)] += F.T @ F
            ginv.append(gi)
            dofs.append(d)
        self.ginv = np.array(ginv)
        self.dofs = np.array(dofs)
        for h in handles:
            A1[2 * h, 2 * h] += weight
            A1[2 * h + 1, 2 * h + 1] += weight
        self.A1 = cho_factor(A1 + 1e-8 * np.eye(2 * n))

        # step 2: every triangle edge should keep its rest length (x and y solve separately)
        e = self.tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        self.ea = e[:, 0]
        self.eb = e[:, 1]
        self.rest_edges = (verts[self.eb] - verts[self.ea]).reshape(-1, 3, 2)
        A2 = np.zeros((n, n))
        np.add.at(A2, (self.ea, self.ea), 1)
        np.add.at(A2, (self.eb, self.eb), 1)
        np.add.at(A2, (self.ea, self.eb), -1)
        np.add.at(A2, (self.eb, self.ea), -1)
        for h in handles:
            A2[h, h] += weight
        self.A2 = cho_factor(A2 + 1e-8 * np.eye(n))

    def solve(self, targets):
        """targets: {vertex: (x, y)} for every handle. returns the bent vertices (n, 2)"""
        n, w = self.n, self.w
        b1 = np.zeros(2 * n)
        for h, (x, y) in targets.items():
            b1[2 * h] = w * x
            b1[2 * h + 1] = w * y
        u = cho_solve(self.A1, b1)

        # each triangle's rotation from step 1 (scale thrown away), applied to its rest edges
        ab = np.einsum("tij,tj->ti", self.ginv[:, :2], u[self.dofs])
        s = np.hypot(ab[:, 0], ab[:, 1])
        s[s == 0] = 1
        c = (ab[:, 0] / s)[:, None]
        sn = (ab[:, 1] / s)[:, None]
        ex = self.rest_edges[..., 0]
        ey = self.rest_edges[..., 1]
        tx = (c * ex - sn * ey).ravel()
        ty = (sn * ex + c * ey).ravel()

        bx = np.zeros(n)
        by = np.zeros(n)
        np.add.at(bx, self.eb, tx)
        np.add.at(bx, self.ea, -tx)
        np.add.at(by, self.eb, ty)
        np.add.at(by, self.ea, -ty)
        for h, (x, y) in targets.items():
            bx[h] += w * x
            by[h] += w * y
        return np.stack([cho_solve(self.A2, bx), cho_solve(self.A2, by)], 1)


# ---------- easing ----------


def ease_spring(t, overshoot=1.9, wobble=2.4):
    """0 -> 1 that shoots a little past 1, wobbles once, and settles"""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    damp = 3.0 / max(0.3, overshoot)
    return 1.0 - math.exp(-damp * t) * math.cos(wobble * math.pi * t)


def ease_out(t):
    """0 -> 1, fast then gentle, never past 1"""
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


# ---------- meshes ----------


def strip_mesh(chain, half_width, cap=0.0, rows=14, cols=5):
    """
    A strip of triangles along a chain of points (knuckle -> fingertip, or elbow -> hand),
    carrying on `cap` past the end for the rounded tip. The first two rows are the
    anchors (stay stuck to you), the last two rows are the bit you're holding, which
    moves as one solid patch so the end keeps its shape. `handle` is its middle.
    """
    chain = np.asarray(chain, np.float64)
    seg = np.hypot(*np.diff(chain, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] + cap

    def along(d):
        i = int(np.clip(np.searchsorted(cum, d, "right") - 1, 0, len(seg) - 1))
        t = (d - cum[i]) / (seg[i] or 1)
        return chain[i] + (chain[i + 1] - chain[i]) * t

    verts = []
    across = []
    for r in range(rows):
        d = r / (rows - 1) * total
        tangent = along(min(total, d + total * 0.05)) - along(max(0, d - total * 0.05))
        tangent /= np.hypot(*tangent) or 1
        normal = np.array([-tangent[1], tangent[0]])
        c = along(d)
        for k in range(cols):
            side = -1 + 2 * k / (cols - 1)
            verts.append(c + normal * half_width * side)
            across.append(side)

    tris = []
    for r in range(rows - 1):
        for k in range(cols - 1):
            v = r * cols + k
            tris += [(v, v + 1, v + cols), (v + 1, v + cols + 1, v + cols)]
    anchors = list(range(2 * cols))
    held = list(range((rows - 2) * cols, rows * cols))
    handle = (rows - 1) * cols + cols // 2
    return np.array(verts), np.array(tris), anchors, held, handle


def thin_toward(deformed, start, end, skip, amount):
    """pull the free vertices toward the start -> end line, so a long stretch gets thinner"""
    if amount <= 0:
        return deformed
    d = np.asarray(end, float) - start
    L2 = float(d @ d)
    if L2 < 1e-6:
        return deformed
    t = ((deformed - start) @ d) / L2
    foot = start + np.outer(t, d)
    out = deformed + (foot - deformed) * amount
    out[skip] = deformed[skip]
    return out


# ---------- cut-outs and fill ----------


def capsule(shape, chain, radius):
    """a fat line along a chain of points"""
    m = np.zeros(shape[:2], np.uint8)
    pts = [tuple(int(v) for v in p) for p in chain]
    r = max(1, int(radius))
    for p in pts:
        cv2.circle(m, p, r, 255, -1)
    for a, b in zip(pts, pts[1:]):
        cv2.line(m, a, b, 255, 2 * r)
    return m


def skin_cutout(frame, chain, radius, tolerance=22):
    """
    Which pixels near a finger are actually finger: inside the landmark capsule and
    the same skin colour as the middle of the finger, so the wall beside it is left out.
    """
    cap = capsule(frame.shape, chain, radius)
    core = cv2.erode(cap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if not core.any():
        core = cap
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    cr0 = np.median(ycc[..., 1][core > 0])
    cb0 = np.median(ycc[..., 2][core > 0])
    skin = (np.abs(ycc[..., 1] - cr0) <= tolerance) & (np.abs(ycc[..., 2] - cb0) <= tolerance)
    out = np.where(skin, cap, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(cv2.morphologyEx(out, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    # if the colour check threw most of it away it probably went wrong, so use the capsule
    if out.sum() < 0.22 * cap.sum():
        return cap
    return out


def soft_capsule(shape, chain, radius):
    """a fat line that fades out at its edges, for a cut-out that blends into your face"""
    m = capsule(shape, chain, radius * 0.75)
    k = max(3, int(radius * 0.5) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def fill_hole(frame, hole):
    """paint over a hole with the colours around it (OpenCV's inpaint), in place"""
    x, y, w, h = cv2.boundingRect(hole)
    if w == 0 or h == 0:
        return
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    frame[y0:y1, x0:x1] = cv2.inpaint(frame[y0:y1, x0:x1], hole[y0:y1, x0:x1], 3, cv2.INPAINT_TELEA)


def tug(frame, center, move, sigma):
    """
    Drag the picture around `center` toward `move`, fading out with distance, in place.
    Capped at 1.5 x sigma: any further and the skin would fold over itself.
    """
    move = np.asarray(move, np.float32)
    length = float(np.hypot(*move))
    if length < 0.5:
        return
    move *= min(1.0, 1.5 * sigma / length)
    H, W = frame.shape[:2]
    reach = 3 * sigma + float(np.hypot(*move))
    x0, y0 = max(0, int(center[0] - reach)), max(0, int(center[1] - reach))
    x1, y1 = min(W, int(center[0] + reach)), min(H, int(center[1] + reach))
    if x1 <= x0 or y1 <= y0:
        return
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    # each pixel looks back toward where its skin came from
    w = np.exp(-((xs - center[0] - move[0]) ** 2 + (ys - center[1] - move[1]) ** 2) / (2 * sigma * sigma))
    mx = (xs - move[0] * w).astype(np.float32)
    my = (ys - move[1] * w).astype(np.float32)
    frame[y0:y1, x0:x1] = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------- drawing ----------


def draw_mesh(frame, texture, alpha, src, dst, tris, shine=0.0, shine_line=None, shine_width=0.0, feather=7):
    """
    Warp each triangle of `texture` from `src` to `dst` and lay it over `frame`, using
    `alpha` (a cut-out in texture space) so only skin gets drawn. Optional soft shine
    along `shine_line` that brightens the skin without going white.
    """
    H, W = frame.shape[:2]
    x0 = max(0, int(dst[:, 0].min()) - 2)
    y0 = max(0, int(dst[:, 1].min()) - 2)
    x1 = min(W, int(dst[:, 0].max()) + 3)
    y1 = min(H, int(dst[:, 1].max()) + 3)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    rw, rh = x1 - x0, y1 - y0
    layer = np.zeros((rh, rw, 3), np.float32)
    a = np.zeros((rh, rw), np.uint8)

    for t in tris:
        s = src[t].astype(np.float32)
        d = (dst[t] - (x0, y0)).astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(d)
        dl = d - np.float32([bx, by])
        cx0, cy0 = max(0, bx), max(0, by)
        cx1, cy1 = min(rw, bx + bw), min(rh, by + bh)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        M = cv2.getAffineTransform(s, dl)
        tex = cv2.warpAffine(texture, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        am = cv2.warpAffine(alpha, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        tm = np.zeros((bh, bw), np.uint8)
        # hard triangle edges inside the mesh, the soft outer edge comes from the cut-out
        cv2.fillConvexPoly(tm, np.round(dl).astype(np.int32), 255, cv2.LINE_8)
        m = np.minimum(tm, am)[cy0 - by : cy1 - by, cx0 - bx : cx1 - bx]
        sel = m > 0
        layer[cy0:cy1, cx0:cx1][sel] = tex[cy0 - by : cy1 - by, cx0 - bx : cx1 - bx][sel]
        np.maximum(a[cy0:cy1, cx0:cx1], m, out=a[cy0:cy1, cx0:cx1])

    if shine > 0.01 and shine_line is not None:
        glow = np.zeros((rh, rw), np.uint8)
        p, q = (np.asarray(v) - (x0, y0) for v in shine_line)
        bw = max(2, int(shine_width))
        cv2.line(glow, tuple(p.astype(int)), tuple(q.astype(int)), 255, bw, cv2.LINE_AA)
        k = max(3, bw | 1)
        band = cv2.GaussianBlur(glow, (k, k), 0).astype(np.float32)[..., None] / 255
        f = 1 + min(0.25, shine) * band
        # never brighter than the brightest channel allows, so the colour stays
        np.minimum(f, 255 / np.maximum(layer.max(axis=2, keepdims=True), 1e-3), out=f)
        layer *= f

    # feather the edge inward so it never picks up the black around the layer
    if feather >= 3:
        a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        a = cv2.GaussianBlur(a, (feather | 1, feather | 1), 0)
    af = (a.astype(np.float32) / 255)[..., None] * (layer.sum(axis=2, keepdims=True) > 0)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - af) + layer * af).astype(np.uint8)
