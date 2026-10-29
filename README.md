# Gomu Gomu

Stretch your face, fingers and arms like Luffy, live on your webcam.

## Running it

```bash
cd ~/Desktop/"One Piece Project"
.venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/python app.py
```

The first run downloads the MediaPipe hand, face and body models into `models/`. On a Mac, allow camera access for your terminal (or VS Code) in System Settings > Privacy & Security > Camera. Press `q` or `esc` to quit.

## How to use it

- **Pinch your cheek** (or the corner of your mouth) with your thumb and index finger and pull your hand away. A strip of your skin stretches out with your hand.
- **Pinch a fingertip of your other hand** and pull. That finger stretches out from its knuckle.
- Or **click and drag** your cheek or a fingertip with the mouse.
- **Hold an arm out straight** for about a second. A yellow ring fills up around your hand, then your forearm and hand shoot outward. Drop your arm to snap it back.
- Let go and it springs back. Fingers and arms bounce a little; cheeks just slide back so nothing swings across your face.

## How it works

- **MediaPipe** (hand, face and pose landmarkers) tracks your hands, 478 points on your face, and your arms every frame.
- Every stretch is the **real picture of your skin bent like rubber**, the method from [gum-gum-hand-stretch](https://github.com/gazellecheetah/gum-gum-hand-stretch), written from scratch here:
  - When you grab, a strip of the picture is cut out: along the finger (trimmed to the finger's own skin colour so the wall beside it is left out), along your forearm, or a strip of cheek from inside your face out to where you grabbed.
  - It's turned into a mesh of triangles and bent with **as-rigid-as-possible** deformation (Igarashi, Moscovich & Hughes 2005, in [rubber.py](rubber.py)). The end near your body stays stuck to you, the end you're holding moves as one solid patch, and the triangles in between stay as close to their real shape as they can, so it never folds over itself.
  - It narrows and gets a soft shine the further past normal length it goes. At normal length it's exactly the real thing again, so it springs back without a seam.
  - The spot where the real finger / arm / cheek strip is gets filled in with OpenCV's inpaint, so you only ever see the stretched one.
  - For a cheek, the skin around where you grabbed also gets dragged toward your hand a little (capped so it can't fold).
- All the tuning numbers are at the top of [app.py](app.py).

## The old browser version

The earlier version that ran in a web page is still here (`index.html`, `main.js`, `rubber.js`, `fruits.js`, `style.css`). Serve the folder with `python3 -m http.server 8123` and open http://localhost:8123 to run it.
