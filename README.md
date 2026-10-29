# Gomu Gomu

Stretch your face and body like Luffy, live on your webcam.

## How to use it

- **Pinch** your cheek, the corner of your mouth, your arm or your neck with your thumb and index finger and pull your hand away.
- **Pinch a fingertip of your other hand** and pull. That finger stretches out from its knuckle like rubber.
- Or **click and drag** any part of yourself with the mouse (or your finger on a phone).
- **Hold an arm out straight** for about a second. A yellow ring fills up around your hand, then your hand shoots outward and your arm stretches. Drop your arm to snap it back.
- A rubbery tube of your skin stretches from where you grabbed to your fingers. Let go and it shoots back into you.

Only you stretch. You can't grab the background.

## How it works

- **MediaPipe Face Landmarker** finds 478 points on your face every frame. Its outline decides what you can grab, and every grab sticks to the nearest point so it moves with your head.
- **MediaPipe Pose Landmarker** tracks your shoulders, arms and torso. A grab on your body sticks to the closest bone, so it follows your arm when you move. The arm doing the pinching is skipped, otherwise it would just grab itself.
- **MediaPipe Image Segmenter** works out which pixels are you and which are background, so you can't grab the wall.
- **MediaPipe Hand Landmarker** tracks both hands. A pinch is when your thumb and index fingertips get close compared with the size of your hand.
- The webcam image is drawn in **WebGL** on an 80x60 grid. A grab only tugs the grid a little around the grab point (capped so the skin never folds over), so your face stays your face.
- The rest of the pull is a **rubber ribbon** drawn on top: a curved strip from the grab point to your fingers, textured with a slice of your skin from where you grabbed so it streaks like stretched rubber. It sags into a curve, tapers toward your fingers, gets thinner the longer it gets, and has dark edges with a soft shine so it reads as a round tube. Holding your arm out turns your forearm and hand into one of these ribbons. Grabbing a fingertip with your other hand does the same with that finger.
- When a finger or arm stretches, the spot where it really is gets filled in with the background, so you only ever see the stretched one. The fill works like OpenCV's inpaint: the camera image is shrunk step by step on the GPU ignoring the hole, then grown back up filling the hole from the smaller copies, so colours from all around blend in smoothly (a "push-pull" fill).
- The ribbon look (curve, taper, thinning with length, shine) was inspired by [gum-gum-hand-stretch](https://github.com/gazellecheetah/gum-gum-hand-stretch).

## Running it

The camera only works on `localhost` or https, so serve the folder rather than opening the file directly:

```bash
cd "One Piece Project"
python3 -m http.server 8000
# open http://localhost:8000
```

No install needed. MediaPipe loads from a CDN.
