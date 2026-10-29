"""
Gum Gum: stretch your face, fingers and arms like Luffy, live on your webcam.

- Pinch your cheek with your thumb and index finger and pull your hand away.
- Pinch a fingertip of your other hand and pull: that finger stretches.
- Or click and drag with the mouse.
- Hold an arm out straight for a second: a ring fills up, then your arm shoots out.
- Let go and it springs back.

    q / esc   quit

The stretch is the real picture of your skin, bent with ARAP (see rubber.py), the
same method https://github.com/gazellecheetah/gum-gum-hand-stretch uses for fingers.

Run:
    .venv/bin/python app.py
"""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import rubber

# ---------- tuning ----------

CAM_WIDTH = 1280
CAM_HEIGHT = 720

# pinch = thumb and index tips closer than this (as a fraction of hand size). a bit of
# hysteresis so it doesn't flicker between pinch and not pinch
PINCH_START = 0.3
PINCH_END = 0.45
# how much of the way the pinch point moves toward your real pinch each frame. lower = smoother
PINCH_SMOOTH = 0.5
# pinch this close to a fingertip of your other hand (fraction of screen height) to grab that finger
FINGER_REACH = 0.07

# a cheek grab is a strip of skin this wide (fraction of face width) ...
CHEEK_SIZE = 0.15
# ... and this many widths long, from inside your face out to where you grabbed
CHEEK_LENGTH = 2.5

# how much a stretch narrows and shines the further past normal length it goes
THIN = 0.15
SHINE = 0.08
# seconds to spring back. fingers and arms bounce a little, cheeks just slide back so
# nothing swings across your face
SNAP_TIME = 0.3
CHEEK_SNAP_TIME = 0.2

# hold an arm out straight for a bit and it shoots out
ARM_CHARGE = 0.8  # seconds of holding before it starts to stretch
ARM_SPEED = 0.5  # how fast it grows, in screen heights per second
ARM_MAX = 0.6
ARM_GRACE = 0.25  # seconds tracking can drop out without cancelling

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# outline of the face in the 478 point mesh, used to check if a grab is on your face
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# each finger from its knuckle to its tip, in the hand landmarks
FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

ARMS = [
    {"shoulder": 11, "elbow": 13, "wrist": 15, "index": 19},
    {"shoulder": 12, "elbow": 14, "wrist": 16, "index": 20},
]

WINDOW = "Gum Gum"


def model_path(name):
    path = os.path.join(MODEL_DIR, name + ".task")
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"downloading the {name} model (first run only)...")
        urllib.request.urlretrieve(MODELS[name], path)
    return path


def dist(a, b):
    return float(np.hypot(*(np.asarray(a, float) - b)))


# ---------- a stretch ----------


class Stretch:
    """
    A bit of you being stretched: a strip of the real picture along `chain` (the first
    point stays stuck to you, the last end is pulled), bent with ARAP. It thins and
    shines the further past normal length it goes, and at normal length it's exactly
    the real thing again, so it springs back without a seam.
    """

    def __init__(self, frame, chain, radius, alpha, hole_radius, spring):
        chain = np.asarray(chain, float)
        self.verts, self.tris, self.anchors, self.held, self.handle = rubber.strip_mesh(chain, radius * 1.3, cap=radius * 0.8)
        self.solver = rubber.ARAP(self.verts, self.tris, self.anchors + self.held)
        self.texture = frame.copy()
        self.alpha = alpha
        self.chain0 = chain
        self.base0 = chain[0]
        self.natural = max(1.0, dist(self.verts[self.handle], self.base0))
        self.radius = radius
        self.hole_radius = hole_radius
        self.spring = spring
        self.delta = np.zeros(2)
        self.target = self.verts[self.handle].copy()
        self.deformed = None
        self.stretch = 0.0
        self.shine = 0.0
        self.released = None
        self.tugs = False  # drag the skin around it along too (cheeks)

    def home(self):
        """where the pulled end would be with nothing stretched, following you"""
        return self.verts[self.handle] + self.delta

    def release(self):
        self.released = (time.monotonic(), self.target.copy())

    def update(self, base, pull=None):
        """
        base: where the stuck-down end is now (None if it couldn't be seen this frame).
        pull: where you're pulling to. returns False once it has sprung back.
        """
        if base is not None:
            self.delta = np.asarray(base, float) - self.base0
        if self.released:
            start_t, start = self.released
            t = (time.monotonic() - start_t) / (SNAP_TIME if self.spring else CHEEK_SNAP_TIME)
            if t >= 1:
                return False
            e = rubber.ease_spring(t) if self.spring else rubber.ease_out(t)
            self.target = start + (self.home() - start) * e
        elif pull is not None:
            self.target = np.asarray(pull, float)

        targets = {a: self.verts[a] + self.delta for a in self.anchors}
        move = self.target - self.verts[self.handle]
        targets.update({v: self.verts[v] + move for v in self.held})
        bent = self.solver.solve(targets)

        # how far past its normal length it is. drives the thinning and the shine, so both
        # fade away by themselves as it comes back
        root = self.base0 + self.delta
        amt = float(np.clip(dist(self.target, root) / self.natural - 1, 0, 2))
        self.deformed = rubber.thin_toward(bent, root, self.target, self.anchors + self.held, THIN * amt)
        self.shine = SHINE * amt
        self.stretch = dist(self.target, self.home()) / self.radius
        return True

    def hole(self, shape):
        # fill in where the real thing is, but only once it has actually moved away, so
        # there's no blurry outline when it's back in place
        if self.stretch < 0.5:
            return None
        return rubber.capsule(shape, self.chain0 + self.delta, self.hole_radius)

    def tug(self, frame):
        # the skin around where you grabbed gets dragged toward your hand a little
        if self.tugs and self.deformed is not None:
            grabbed = self.chain0[-1] + self.delta
            rubber.tug(frame, grabbed, self.target - self.home(), self.radius * 0.8)

    def draw(self, frame):
        if self.deformed is None:
            return
        root = self.base0 + self.delta
        d = self.target - root
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)
        side = n * self.radius * 0.3  # the shine sits a little off center
        rubber.draw_mesh(
            frame, self.texture, self.alpha, self.verts, self.deformed, self.tris,
            shine=self.shine, shine_line=(root + side, self.target + side), shine_width=self.radius * 0.55,
        )


# ---------- the app ----------


class App:
    def __init__(self):
        def opts(cls, name, **kw):
            return cls(base_options=BaseOptions(model_asset_path=model_path(name)), running_mode=vision.RunningMode.VIDEO, **kw)

        self.hand_model = vision.HandLandmarker.create_from_options(opts(vision.HandLandmarkerOptions, "hand", num_hands=2))
        self.face_model = vision.FaceLandmarker.create_from_options(opts(vision.FaceLandmarkerOptions, "face", num_faces=1))
        self.pose_model = vision.PoseLandmarker.create_from_options(opts(vision.PoseLandmarkerOptions, "pose", num_poses=1))

        self.frame = None
        self.hands = []  # each (21, 2) in pixels
        self.face = None  # (478, 2)
        self.pose = None  # (33, 3): x, y, visibility
        self.last_ts = 0

        self.grabs = {}  # pointer id -> grab
        self.springing = []  # grabs you let go of, springing back
        self.pinching = [False, False]
        self.smoothed = [None, None]
        self.arms = [{"hold": 0.0, "lost": 0.0, "grab": None} for _ in ARMS]
        self.mouse_down = False

    # ----- tracking -----

    def detect(self, frame):
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # timestamps have to keep going up
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        px = lambda lms: np.array([(p.x * w, p.y * h) for p in lms])

        self.hands = [px(lms) for lms in self.hand_model.detect_for_video(image, ts).hand_landmarks]
        faces = self.face_model.detect_for_video(image, ts).face_landmarks
        self.face = px(faces[0]) if faces else None
        poses = self.pose_model.detect_for_video(image, ts).pose_landmarks
        self.pose = np.array([(p.x * w, p.y * h, p.visibility or 0) for p in poses[0]]) if poses else None

    def base_of(self, g):
        """where the stuck-down end of a grab is this frame, or None if it can't be seen"""
        if g["kind"] == "finger":
            # hands can swap order between frames, so find the grabbed one by where its knuckle was
            k = g["finger"][0]
            others = [hand for i, hand in enumerate(self.hands) if i != g["grabber"]]
            if not others:
                return None
            hand = min(others, key=lambda hand: dist(hand[k], g["knuckle"]))
            g["knuckle"] = hand[k]
            return hand[k]
        if g["kind"] == "cheek":
            if self.face is None:
                return None
            return g["stretch"].base0 + self.face[g["landmark"]] - g["landmark0"]
        if g["kind"] == "arm":
            p = self.pose[g["arm"]["elbow"]] if self.pose is not None else None
            return p[:2] if p is not None and p[2] > 0.5 else None

    # ----- grabbing -----

    def start_grab(self, gid, p, grabber=None):
        h = self.frame.shape[0]

        # right on a fingertip of the other hand: stretch that finger
        best = None
        for i, hand in enumerate(self.hands):
            if i == grabber:
                continue
            for f in FINGERS:
                d = dist(hand[f[-1]], p)
                if d < FINGER_REACH * h and (best is None or d < best[0]):
                    best = (d, hand, f)
        if best:
            _, hand, f = best
            chain = hand[f]
            # finger width: roughly the gap between neighbouring knuckles
            width = np.mean([dist(hand[a], hand[b]) for a, b in [(5, 9), (9, 13), (13, 17)]])
            radius = width * 0.95 / 2
            s = Stretch(self.frame, chain, radius, rubber.skin_cutout(self.frame, chain, radius), radius * 1.3, spring=True)
            # start the pull exactly where the fingertip is, so it doesn't jump to your pinch
            self.grabs[gid] = {
                "kind": "finger", "stretch": s, "finger": f, "knuckle": chain[0], "grabber": grabber,
                "offset": s.home() - p, "pull": np.asarray(p, float),
            }
            return

        # on your face: the cheek gets made into a stretch once you start pulling
        if self.face is not None:
            oval = self.face[FACE_OVAL].astype(np.float32)
            if cv2.pointPolygonTest(oval, (float(p[0]), float(p[1])), False) >= 0:
                lm = int(np.argmin(np.hypot(*(self.face - p).T)))
                self.grabs[gid] = {
                    "kind": "cheek", "stretch": None, "p0": np.asarray(p, float), "landmark": lm,
                    "landmark0": self.face[lm].copy(), "radius": dist(self.face[454], self.face[234]) * CHEEK_SIZE,
                    "grabber": grabber, "pull": np.asarray(p, float),
                }

    def end_grab(self, gid):
        g = self.grabs.pop(gid, None)
        if g and g["stretch"]:
            g["stretch"].release()
            self.springing.append(g)

    def update_cheek(self, g):
        if g["stretch"] is None:
            # wait until you've pulled a little, so we know which way the strip goes
            p0 = g["p0"] + (self.face[g["landmark"]] - g["landmark0"] if self.face is not None else 0)
            pull = g["pull"] - p0
            if np.hypot(*pull) < g["radius"] * 0.5:
                return
            d = pull / np.hypot(*pull)
            r = g["radius"]
            # the strip's rounded end reaches 0.8 widths past its last point, so stop short
            # by that much: the bit you're holding is the skin you grabbed, not what's past it
            chain = [p0 - d * r * (CHEEK_LENGTH + 0.8), p0 - d * r * 0.8]
            g["stretch"] = Stretch(self.frame, chain, r, rubber.soft_capsule(self.frame.shape, chain, r), r * 0.9, spring=False)
            g["stretch"].tugs = True
            g["landmark0"] = self.face[g["landmark"]].copy()
            g["offset"] = g["stretch"].home() - p0
        g["stretch"].update(self.base_of(g), g["pull"] + g["offset"])

    # ----- hands -----

    def update_hands(self):
        seen = [False, False]
        for h, hand in enumerate(self.hands[:2]):
            seen[h] = True
            size = dist(hand[0], hand[9])
            gap = dist(hand[4], hand[8])
            p = (hand[4] + hand[8]) / 2
            pinch = gap < size * (PINCH_END if self.pinching[h] else PINCH_START)
            # smooth the pinch point so the stretch doesn't jitter
            if pinch and self.pinching[h] and self.smoothed[h] is not None:
                p = self.smoothed[h] + (p - self.smoothed[h]) * PINCH_SMOOTH
            self.smoothed[h] = p

            gid = f"h{h}"
            if pinch and not self.pinching[h]:
                self.start_grab(gid, p, grabber=h)
            elif pinch and gid in self.grabs:
                self.grabs[gid]["pull"] = p
            elif not pinch and self.pinching[h]:
                self.end_grab(gid)
            self.pinching[h] = pinch

        # hand went off screen mid pinch
        for h in range(2):
            if not seen[h] and self.pinching[h]:
                self.end_grab(f"h{h}")
                self.pinching[h] = False

    def on_mouse(self, event, x, y, flags, _):
        if self.frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.start_grab("mouse", np.array([x, y], float))
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down and "mouse" in self.grabs:
            self.grabs["mouse"]["pull"] = np.array([x, y], float)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            self.end_grab("mouse")

    # ----- arms -----

    def arm_out(self, arm):
        """which way the arm points if it's held out straight, else None"""
        if self.pose is None:
            return None
        S, E, W = (self.pose[arm[k]] for k in ("shoulder", "elbow", "wrist"))
        if min(S[2], E[2], W[2]) < 0.5:
            return None
        upper = E[:2] - S[:2]
        fore = W[:2] - E[:2]
        lu, lf = np.hypot(*upper), np.hypot(*fore)
        if not lu or not lf:
            return None
        # the elbow is straight when the upper arm and forearm point the same way
        straight = upper @ fore / (lu * lf)
        d = upper + fore
        d = d / np.hypot(*d)
        # an arm hanging down by your side is straight too, so skip those
        if straight < 0.9 or d[1] > 0.5:
            return None
        return d, lf

    def update_arms(self, dt):
        H = self.frame.shape[0]
        for arm, st in zip(ARMS, self.arms):
            out = self.arm_out(arm)
            if out is None:
                st["lost"] += dt
                if st["lost"] > ARM_GRACE:
                    st["hold"] = 0.0
                    if st["grab"]:
                        st["grab"]["stretch"].release()
                        self.springing.append(st["grab"])
                        st["grab"] = None
                continue
            st["lost"] = 0.0
            st["hold"] += dt

            d, fore = out
            ext = min(ARM_MAX, max(0.0, st["hold"] - ARM_CHARGE) * ARM_SPEED) * H
            if ext > 0 and not st["grab"]:
                chain = self.pose[[arm["elbow"], arm["wrist"], arm["index"]], :2]
                r = fore * 0.22
                s = Stretch(self.frame, chain, r, rubber.skin_cutout(self.frame, chain, r), r * 1.5, spring=True)
                st["grab"] = {"kind": "arm", "arm": arm, "stretch": s}
            if st["grab"]:
                s = st["grab"]["stretch"]
                base = self.base_of(st["grab"])
                if base is not None:
                    s.delta = base - s.base0
                s.update(base, s.home() + d * ext)

    def draw_charge(self, out):
        # a ring fills up around your hand while you hold your arm out
        if self.pose is None:
            return
        H = out.shape[0]
        for arm, st in zip(ARMS, self.arms):
            if st["hold"] < 0.15 or st["hold"] > ARM_CHARGE:
                continue
            c = (self.pose[arm["wrist"], :2] + self.pose[arm["index"], :2]) / 2
            r = int(H * 0.05)
            cv2.ellipse(out, tuple(c.astype(int)), (r, r), -90, 0, 360 * st["hold"] / ARM_CHARGE, (48, 196, 244), max(2, int(H * 0.012)), cv2.LINE_AA)

    # ----- main loop -----

    def process(self, frame, dt):
        """one camera frame in, the stretched picture out"""
        self.frame = frame
        self.detect(frame)
        self.update_hands()
        for g in list(self.grabs.values()):
            if g["kind"] == "cheek":
                self.update_cheek(g)
            else:
                base = self.base_of(g)
                if base is None and g["kind"] == "finger":
                    # lost the grabbed hand, let go
                    gid = next(k for k, v in self.grabs.items() if v is g)
                    self.end_grab(gid)
                    continue
                g["stretch"].update(base, g["pull"] + g["offset"])
        self.update_arms(dt)
        self.springing = [g for g in self.springing if g["stretch"].update(self.base_of(g))]

        stretches = [g["stretch"] for g in list(self.grabs.values()) + self.springing if g["stretch"]]
        stretches += [st["grab"]["stretch"] for st in self.arms if st["grab"]]

        out = frame.copy()
        # drag the face around cheek grabs, fill in where the real thing is, then draw the
        # stretched copies on top
        for s in stretches:
            s.tug(out)
        for s in stretches:
            hole = s.hole(out.shape)
            if hole is not None:
                rubber.fill_hole(out, hole)
        for s in stretches:
            s.draw(out)
        self.draw_charge(out)
        return out

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        ok, frame = cam.read()
        if not ok:
            print("can't open the camera. on a Mac, allow camera access for your terminal / VS Code in")
            print("System Settings > Privacy & Security > Camera, then run it again.")
            return 1

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, frame.shape[1], frame.shape[0])
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        last = time.monotonic()

        while True:
            ok, frame = cam.read()
            if not ok:
                break
            now = time.monotonic()
            out = self.process(cv2.flip(frame, 1), min(0.1, now - last))  # mirrored, like a mirror
            last = now

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        cam.release()
        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    sys.exit(App().run())
