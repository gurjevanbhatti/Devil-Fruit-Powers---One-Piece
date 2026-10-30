# Gomu Gomu

Stretch your face and body like Luffy, live on your webcam.

## How to use it

- **Pinch** your cheek, the corner of your mouth, your arm or your neck with your thumb and index finger and pull your hand away.
- Or **click and drag** any part of yourself with the mouse (or your finger on a phone).
- **Hold an arm out straight** for about a second. A yellow ring fills up around your hand, then your hand shoots outward and your arm stretches. Drop your arm to snap it back.
- Let go and it slides smoothly back to normal.
- Press **H** to show your hand tracking (lines and dots over each hand). Press it again to hide it.

Only you stretch. You can't grab the background.

## How it works

- **MediaPipe Face Landmarker** finds 478 points on your face every frame. Its outline decides what you can grab, and every grab sticks to the nearest point so it moves with your head.
- **MediaPipe Pose Landmarker** tracks your shoulders, arms and torso. A grab on your body sticks to the closest bone, so it follows your arm when you move. The arm doing the pinching is skipped, otherwise it would just grab itself.
- **MediaPipe Image Segmenter** works out which pixels are you and which are background, so you can't grab the wall.
- **MediaPipe Hand Landmarker** tracks both hands. A pinch is when your thumb and index fingertips get close compared with the size of your hand.
- The webcam image is drawn in **WebGL** on an 80x60 grid. Pulling moves the grid points with a Gaussian falloff around the grab point, and each point eases toward where it should be, so it slides smoothly back to normal when you let go. The pull radius is small, so the skin stretches into a thin band like the live-action cheek pull, and the more stretched skin is drawn on top.
- An arm stretch is drawn as its own layer on top, with every pixel the segmenter says is background cut out, so your arm stretches but the background behind it stays still. Underneath it, your real forearm and hand get filled in with the background around them (a GPU version of OpenCV's inpaint), so you don't see your real hand behind the stretched one.

## Running it

The camera only works on `localhost` or https, so serve the folder rather than opening the file directly:

```bash
cd "One Piece Project"
python3 -m http.server 8000
# open http://localhost:8000
```

No install needed. MediaPipe loads from a CDN.
