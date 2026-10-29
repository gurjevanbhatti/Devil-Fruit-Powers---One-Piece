import {
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
  ImageSegmenter,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
import { ARAP, easeSpring } from './arap.js'

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
const SEG_MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'

// outline of the face from the 478 point mesh, used to check if a grab is on the face
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

// pose skeleton bones you can grab onto (shoulders, arms, torso)
const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 19], [16, 20], [11, 23], [12, 24], [23, 24],
]
const LEFT_ARM = [13, 15, 17, 19, 21]
const RIGHT_ARM = [14, 16, 18, 20, 22]

// grid resolution, more = smoother stretch
const COLS = 80
const ROWS = 60

// how much of the way to its target the face moves each frame while you pull
const FOLLOW = 0.2
// and once you let go
const SNAP = 0.5
// the face itself only gets tugged this far (in grab radii). any further and the skin
// folds over and you see it twice, so the rest of the pull is drawn as a rubber ribbon
const TUG = 1.2
// how fast a ribbon shoots back when you let go, and how smooth it is
const RIBBON_SNAP = 0.45
const RIBBON_SEGS = 24
// pinch near a fingertip of your other hand (this close, in screen heights) to stretch that finger
const FINGER_REACH = 0.07
// how much of the way the pinch point moves toward your real pinch each frame. lower = smoother
const PINCH_SMOOTH = 0.5

const video = document.getElementById('cam')
const glCanvas = document.getElementById('gl')
const hud = document.getElementById('hud')
const ctx = hud.getContext('2d')
const startScreen = document.getElementById('start')
const statusText = document.getElementById('status')

let face = null // landmarks in screen space (mirrored), or null
let pose = null
let faceLandmarker = null
let handLandmarker = null
let poseLandmarker = null
let segmenter = null
let aspect = 4 / 3

// everything is in 0..1 screen coords, x is mirrored so it acts like a mirror
const grabs = new Map() // id -> { attach, x, y, sigma }
let shout = { text: '', t: 0 }

// ---------- webgl setup ----------

const gl = glCanvas.getContext('webgl', { premultipliedAlpha: false })

const vertSrc = `
attribute vec2 rest;
attribute vec2 offset;
varying vec2 uv;
void main() {
  vec2 p = rest + offset;
  uv = vec2(1.0 - rest.x, rest.y);
  // the further skin is pulled the closer it gets drawn, so the stretched bit goes over your face
  float z = -clamp(length(offset), 0.0, 0.99);
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, z, 1.0);
}`

const fragSrc = `
precision mediump float;
uniform sampler2D tex;
varying vec2 uv;
void main() {
  gl_FragColor = texture2D(tex, uv);
}`

function compile(type, src) {
  const s = gl.createShader(type)
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s))
  return s
}

const prog = gl.createProgram()
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertSrc))
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragSrc))
gl.linkProgram(prog)
gl.useProgram(prog)

const vertCount = (COLS + 1) * (ROWS + 1)
const rest = new Float32Array(vertCount * 2)
const offset = new Float32Array(vertCount * 2)

for (let r = 0; r <= ROWS; r++) {
  for (let c = 0; c <= COLS; c++) {
    const i = (r * (COLS + 1) + c) * 2
    rest[i] = c / COLS
    rest[i + 1] = r / ROWS
  }
}

const indices = new Uint16Array(COLS * ROWS * 6)
let k = 0
for (let r = 0; r < ROWS; r++) {
  for (let c = 0; c < COLS; c++) {
    const a = r * (COLS + 1) + c
    const b = a + 1
    const d = a + COLS + 1
    const e = d + 1
    indices.set([a, b, d, b, e, d], k)
    k += 6
  }
}

function attrib(name, data, usage) {
  const buf = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  gl.bufferData(gl.ARRAY_BUFFER, data, usage)
  const loc = gl.getAttribLocation(prog, name)
  gl.enableVertexAttribArray(loc)
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
  return buf
}

const restBuf = attrib('rest', rest, gl.STATIC_DRAW)
const offsetBuf = attrib('offset', offset, gl.DYNAMIC_DRAW)
const restLoc = gl.getAttribLocation(prog, 'rest')
const offsetLoc = gl.getAttribLocation(prog, 'offset')

const indexBuf = gl.createBuffer()
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuf)
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

gl.enable(gl.DEPTH_TEST)
gl.depthFunc(gl.LEQUAL)

const tex = gl.createTexture()
gl.bindTexture(gl.TEXTURE_2D, tex)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

// ---------- ribbon program ----------

// the stretched skin is a strip drawn on top, textured with the skin where you grabbed
const ribbonVert = `
attribute vec2 pos;
attribute vec2 src;
attribute float shade;
varying vec2 uv;
varying float vShade;
void main() {
  uv = vec2(1.0 - src.x, src.y);
  vShade = shade;
  gl_Position = vec4(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
}`

const ribbonFrag = `
precision mediump float;
uniform sampler2D tex;
varying vec2 uv;
varying float vShade;
void main() {
  gl_FragColor = vec4(texture2D(tex, uv).rgb * vShade, 1.0);
}`

const ribbonProg = gl.createProgram()
gl.attachShader(ribbonProg, compile(gl.VERTEX_SHADER, ribbonVert))
gl.attachShader(ribbonProg, compile(gl.FRAGMENT_SHADER, ribbonFrag))
gl.linkProgram(ribbonProg)
const ribbonLocs = ['pos', 'src', 'shade'].map((n) => gl.getAttribLocation(ribbonProg, n))
const ribbonBuf = gl.createBuffer()

// ---------- erase (fill) ----------

// paints over where a stretched finger or arm really is, so you only see the stretched one.
// the hole is filled like OpenCV's inpaint: the camera image is shrunk step by step
// ignoring the hole, then grown back up filling the hole from the smaller copies, so
// the colours all around it blend in from every side ("push-pull" fill)
const FILL_WIDTH = 320 // the fill is smooth anyway, so it's worked out on a small copy

// everything here draws shapes given in screen coords (0..1, y down)
const fillVert = `
attribute vec2 pos;
varying vec2 sc;
void main() {
  sc = pos;
  gl_Position = vec4(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
}`

const fillShaders = {
  // the camera image, as you see it on screen
  copy: `
precision mediump float;
uniform sampler2D src;
varying vec2 sc;
void main() {
  gl_FragColor = vec4(texture2D(src, vec2(1.0 - sc.x, sc.y)).rgb, 1.0);
}`,
  // cuts the hole out (alpha 0 = unknown)
  hole: `
precision mediump float;
void main() {
  gl_FragColor = vec4(0.0);
}`,
  // half size: one linear sample between 4 pixels averages them. colours are
  // multiplied by alpha, so the hole doesn't darken the average
  down: `
precision mediump float;
uniform sampler2D src;
uniform vec2 srcSize;
void main() {
  gl_FragColor = texture2D(src, gl_FragCoord.xy * 2.0 / srcSize);
}`,
  // the smallest copy, turned back into plain colour
  top: `
precision mediump float;
uniform sampler2D src;
uniform vec2 size;
void main() {
  vec4 c = texture2D(src, gl_FragCoord.xy / size);
  gl_FragColor = vec4(c.rgb / max(c.a, 0.004), 1.0);
}`,
  // keep what's known at this size, fill the rest from the filled copy one size smaller
  up: `
precision mediump float;
uniform sampler2D src;
uniform sampler2D coarse;
uniform vec2 size;
void main() {
  vec4 f = texture2D(src, gl_FragCoord.xy / size);
  vec3 c = texture2D(coarse, gl_FragCoord.xy / size).rgb;
  gl_FragColor = vec4(f.rgb + (1.0 - f.a) * c, 1.0);
}`,
  // finally, paint the filled image over the hole on screen
  paint: `
precision mediump float;
uniform sampler2D src;
varying vec2 sc;
void main() {
  gl_FragColor = vec4(texture2D(src, vec2(sc.x, 1.0 - sc.y)).rgb, 1.0);
}`,
}

const fill = {}
for (const [name, frag] of Object.entries(fillShaders)) {
  const pr = gl.createProgram()
  gl.attachShader(pr, compile(gl.VERTEX_SHADER, fillVert))
  gl.attachShader(pr, compile(gl.FRAGMENT_SHADER, frag))
  gl.linkProgram(pr)
  const uniforms = {}
  for (const u of ['src', 'coarse', 'srcSize', 'size']) uniforms[u] = gl.getUniformLocation(pr, u)
  fill[name] = { prog: pr, pos: gl.getAttribLocation(pr, 'pos'), u: uniforms }
}
const fillBuf = gl.createBuffer()
const screenQuad = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]

let pyramid = null // [{ w, h, push: {tex, fb}, filled: {tex, fb} }] biggest first

function makeTarget(w, h) {
  const t = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, t)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
  const fb = gl.createFramebuffer()
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0)
  return { tex: t, fb }
}

function buildPyramid() {
  pyramid = []
  let w = FILL_WIDTH
  let h = Math.max(1, Math.round(FILL_WIDTH / aspect))
  for (;;) {
    pyramid.push({ w, h, push: makeTarget(w, h), filled: makeTarget(w, h) })
    if (w === 1 && h === 1) break
    w = Math.max(1, Math.ceil(w / 2))
    h = Math.max(1, Math.ceil(h / 2))
  }
  pyramid.aspect = aspect
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
}

function fillPass(name, verts, target, level, setup) {
  const f = fill[name]
  gl.useProgram(f.prog)
  gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null)
  if (target) gl.viewport(0, 0, level.w, level.h)
  else gl.viewport(0, 0, glCanvas.width, glCanvas.height)
  setup?.(f.u)
  gl.bindBuffer(gl.ARRAY_BUFFER, fillBuf)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STREAM_DRAW)
  gl.enableVertexAttribArray(f.pos)
  gl.vertexAttribPointer(f.pos, 2, gl.FLOAT, false, 0, 0)
  gl.drawArrays(gl.TRIANGLES, 0, verts.length / 2)
  gl.disableVertexAttribArray(f.pos)
}

function bindTex(unit, t) {
  gl.activeTexture(gl.TEXTURE0 + unit)
  gl.bindTexture(gl.TEXTURE_2D, t)
}

// ---------- sizing ----------

function resize() {
  aspect = video.videoWidth / video.videoHeight || 4 / 3
  let w = innerWidth
  let h = w / aspect
  if (h > innerHeight) {
    h = innerHeight
    w = h * aspect
  }
  const dpr = devicePixelRatio || 1
  for (const c of [glCanvas, hud]) {
    c.width = Math.round(w * dpr)
    c.height = Math.round(h * dpr)
    c.style.width = w + 'px'
    c.style.height = h + 'px'
  }
  gl.viewport(0, 0, glCanvas.width, glCanvas.height)
}
addEventListener('resize', resize)

// ---------- face helpers ----------

function insideFace(x, y) {
  if (!face) return false
  // ray casting against the face outline
  let inside = false
  for (let i = 0, j = FACE_OVAL.length - 1; i < FACE_OVAL.length; j = i++) {
    const a = face[FACE_OVAL[i]]
    const b = face[FACE_OVAL[j]]
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside
  }
  return inside
}

function nearestLandmark(x, y) {
  let best = 0
  let bestD = Infinity
  for (let i = 0; i < face.length; i++) {
    const d = ((face[i].x - x) * aspect) ** 2 + (face[i].y - y) ** 2
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return best
}

function faceWidth() {
  if (!face) return 0.3
  return Math.abs(face[454].x - face[234].x) * aspect
}

// ---------- grabbing ----------

function onPerson(x, y) {
  // asks the segmenter if this pixel is you or the background
  const res = segmenter.segmentForVideo(video, performance.now())
  const mask = res.categoryMask
  const mx = Math.min(mask.width - 1, Math.floor((1 - x) * mask.width))
  const my = Math.min(mask.height - 1, Math.floor(y * mask.height))
  const category = mask.getAsUint8Array()[my * mask.width + mx]
  res.close()
  return category !== 0 // 0 is background
}

function nearestBone(x, y, skip) {
  if (!pose) return null
  let best = null
  let bestD = Infinity
  for (const [a, b] of BONES) {
    if (skip.includes(a) || skip.includes(b)) continue
    const A = pose[a]
    const B = pose[b]
    if (A.visibility < 0.5 || B.visibility < 0.5) continue
    // closest point on the bone (with aspect fixed so the distances are real)
    const bx = (B.x - A.x) * aspect
    const by = B.y - A.y
    const px = (x - A.x) * aspect
    const py = y - A.y
    const t = Math.max(0, Math.min(1, (px * bx + py * by) / (bx * bx + by * by || 1)))
    const d = (px - bx * t) ** 2 + (py - by * t) ** 2
    if (d < bestD) {
      bestD = d
      best = { a, b, t }
    }
  }
  return best
}

function bonePoint({ a, b, t }) {
  return [pose[a].x + (pose[b].x - pose[a].x) * t, pose[a].y + (pose[b].y - pose[a].y) * t]
}

// the arm doing the pinching shouldn't be grabbable by itself
function grabbingArm(wrist) {
  if (!pose || !wrist) return []
  const dl = Math.hypot(pose[15].x - wrist.x, pose[15].y - wrist.y)
  const dr = Math.hypot(pose[16].x - wrist.x, pose[16].y - wrist.y)
  return dl < dr ? LEFT_ARM : RIGHT_ARM
}

function startGrab(id, x, y, wrist) {
  const onFace = insideFace(x, y)
  const anyModel = faceLandmarker || segmenter

  // only you are made of rubber, not the background
  if (anyModel && !onFace && !(segmenter && onPerson(x, y))) return

  let attach
  if (onFace) {
    const lm = nearestLandmark(x, y)
    attach = { kind: 'face', lm, ox: x - face[lm].x, oy: y - face[lm].y }
  } else {
    const bone = nearestBone(x, y, grabbingArm(wrist))
    if (bone) {
      const [bx, by] = bonePoint(bone)
      attach = { kind: 'bone', bone, ox: x - bx, oy: y - by }
    } else {
      attach = { kind: 'fixed', ox: x, oy: y }
    }
  }

  // small radius = a thin band of skin like the live action cheek pull
  const sigma = faceWidth() * (onFace ? 0.11 : 0.14)
  grabs.set(id, { attach, x, y, sigma })
}

function moveGrab(id, x, y) {
  const g = grabs.get(id)
  if (g) {
    g.x = x
    g.y = y
  }
}

function endGrab(id) {
  const g = grabs.get(id)
  if (!g) return
  grabs.delete(id)
  // the ribbon keeps going on its own for a moment while it shoots back
  if (g.ribbon) snaps.push(g.ribbon)
  // a stretched finger springs back to where the real one is
  if (g.attach.kind === 'finger' && g.attach.finger.def) {
    const a = g.attach
    fingerSnaps.push({ a, from: a.finger.target, start: performance.now() })
  }
}

function anchor(g) {
  // the grab point sticks to your face / arm so it follows you when you move
  const a = g.attach
  if (a.kind === 'face' && face) return [face[a.lm].x + a.ox, face[a.lm].y + a.oy]
  if (a.kind === 'bone' && pose) {
    const [bx, by] = bonePoint(a.bone)
    return [bx + a.ox, by + a.oy]
  }
  if (a.kind === 'arm' && pose) return handPoint(a.arm)
  if (a.kind === 'finger') return a.mcp
  if (a.kind === 'fixed') return [a.ox, a.oy]
  return [g.x, g.y] // lost tracking, just go slack
}

function toScreen(e) {
  const r = glCanvas.getBoundingClientRect()
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]
}

glCanvas.addEventListener('pointerdown', (e) => {
  glCanvas.setPointerCapture(e.pointerId)
  startGrab('p' + e.pointerId, ...toScreen(e))
})
glCanvas.addEventListener('pointermove', (e) => moveGrab('p' + e.pointerId, ...toScreen(e)))
glCanvas.addEventListener('pointerup', (e) => endGrab('p' + e.pointerId))
glCanvas.addEventListener('pointercancel', (e) => endGrab('p' + e.pointerId))

// ---------- hand pinching ----------

const pinching = [false, false]
const smoothed = [null, null]
let pinchPoints = []
let hands = [] // every hand this frame, mirrored to screen space

// fingertip -> its knuckle, the stretched finger roots at the knuckle
const FINGERS = [
  { tipIdx: 8, mcpIdx: 5, chain: [5, 6, 7, 8] },
  { tipIdx: 12, mcpIdx: 9, chain: [9, 10, 11, 12] },
  { tipIdx: 16, mcpIdx: 13, chain: [13, 14, 15, 16] },
  { tipIdx: 20, mcpIdx: 17, chain: [17, 18, 19, 20] },
]

function fingerWidth(hand) {
  // roughly the gap between neighbouring knuckles
  let sum = 0
  for (const [a, b] of [[5, 9], [9, 13], [13, 17]]) sum += Math.hypot((hand[a].x - hand[b].x) * aspect, hand[a].y - hand[b].y)
  return sum / 3
}

function grabFinger(id, x, y, h) {
  // is this pinch right on a fingertip of your other hand?
  let best = null
  let bestD = FINGER_REACH
  hands.forEach((hand, o) => {
    if (o === h) return
    for (const f of FINGERS) {
      const d = Math.hypot((hand[f.tipIdx].x - x) * aspect, hand[f.tipIdx].y - y)
      if (d < bestD) {
        bestD = d
        best = { hand, f }
      }
    }
  })
  if (!best) return false
  const { hand, f } = best
  const attach = {
    kind: 'finger',
    ...f,
    hand,
    mcp: [hand[f.mcpIdx].x, hand[f.mcpIdx].y],
    width: fingerWidth(hand),
  }
  attach.finger = buildFinger(attach)
  if (!attach.finger) return false
  // start the pull exactly where the fingertip is, so it doesn't jump to your pinch
  const [tx, ty] = attach.finger.rest[attach.finger.tipV]
  attach.pinchOffset = [tx - x * aspect, ty - y]
  grabs.set(id, { attach, x, y, sigma: 0 })
  return true
}

function trackFinger(a, grabber) {
  // hands can swap order between frames, so find the grabbed finger by where its knuckle was
  let best = null
  let bestD = Infinity
  hands.forEach((hand, o) => {
    if (o === grabber) return
    const d = Math.hypot(hand[a.mcpIdx].x - a.mcp[0], hand[a.mcpIdx].y - a.mcp[1])
    if (d < bestD) {
      bestD = d
      best = hand
    }
  })
  if (!best) return
  a.hand = best
  a.mcp = [best[a.mcpIdx].x, best[a.mcpIdx].y]
  a.width = fingerWidth(best)
}

function updateHands(result) {
  pinchPoints = []
  const seen = [false, false]
  hands = result.landmarks.map((hand) => hand.map((p) => ({ x: 1 - p.x, y: p.y })))

  result.landmarks.forEach((hand, h) => {
    if (h > 1) return
    seen[h] = true
    const thumb = hand[4]
    const index = hand[8]
    const size = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y)
    const gap = Math.hypot(thumb.x - index.x, thumb.y - index.y)
    let x = 1 - (thumb.x + index.x) / 2
    let y = (thumb.y + index.y) / 2

    // a bit of hysteresis so it doesn't flicker between pinch and not pinch
    const isPinch = pinching[h] ? gap < size * 0.45 : gap < size * 0.3
    pinchPoints.push({ x, y, isPinch })
    const wrist = { x: 1 - hand[0].x, y: hand[0].y }

    // smooth the pinch point so the stretch doesn't jitter
    if (isPinch && pinching[h] && smoothed[h]) {
      x = smoothed[h][0] + (x - smoothed[h][0]) * PINCH_SMOOTH
      y = smoothed[h][1] + (y - smoothed[h][1]) * PINCH_SMOOTH
    }
    smoothed[h] = [x, y]

    const id = 'h' + h
    if (isPinch && !pinching[h]) {
      if (!grabFinger(id, x, y, h)) startGrab(id, x, y, wrist)
    } else if (isPinch) {
      moveGrab(id, x, y)
      const g = grabs.get(id)
      if (g?.attach.kind === 'finger') trackFinger(g.attach, h)
    } else if (pinching[h]) endGrab(id)
    pinching[h] = isPinch
  })

  // hand went off screen mid pinch
  for (let h = 0; h < 2; h++) {
    if (!seen[h] && pinching[h]) {
      endGrab('h' + h)
      pinching[h] = false
    }
  }
}

// ---------- arm stretch ----------

// hold an arm out straight for a bit and the hand shoots out like rubber
const ARMS = [
  { id: 'armL', shoulder: 11, elbow: 13, wrist: 15, index: 19 },
  { id: 'armR', shoulder: 12, elbow: 14, wrist: 16, index: 20 },
]
const ARM_CHARGE = 0.8 // seconds of holding before it starts to stretch
const ARM_SPEED = 0.5 // how fast it grows, in screen heights per second
const ARM_MAX = 0.6
const ARM_GRACE = 0.25 // seconds tracking can drop out without cancelling
const armState = ARMS.map(() => ({ hold: 0, lost: 0, hand: null }))

function handPoint(arm) {
  const W = pose[arm.wrist]
  const I = pose[arm.index]
  return [(W.x + I.x) / 2, (W.y + I.y) / 2]
}

function armOut(arm) {
  if (!pose) return null
  const S = pose[arm.shoulder]
  const E = pose[arm.elbow]
  const W = pose[arm.wrist]
  if (S.visibility < 0.5 || E.visibility < 0.5 || W.visibility < 0.5) return null
  const ux = (E.x - S.x) * aspect
  const uy = E.y - S.y
  const fx = (W.x - E.x) * aspect
  const fy = W.y - E.y
  const upper = Math.hypot(ux, uy)
  const fore = Math.hypot(fx, fy)
  if (!upper || !fore) return null
  // the elbow is straight when the upper arm and forearm point the same way
  const straight = (ux * fx + uy * fy) / (upper * fore)
  const dx = ux + fx
  const dy = uy + fy
  const len = Math.hypot(dx, dy)
  // an arm hanging down by your side is straight too, so skip those
  if (straight < 0.9 || dy / len > 0.5) return null
  return { dx: dx / len, dy: dy / len, fore }
}

function updateArms(dt) {
  ARMS.forEach((arm, i) => {
    const st = armState[i]
    const out = armOut(arm)
    st.hand = pose ? handPoint(arm) : null

    if (!out) {
      st.lost += dt
      if (st.lost > ARM_GRACE) {
        st.hold = 0
        endGrab(arm.id)
      }
      return
    }
    st.lost = 0
    st.hold += dt

    const ext = Math.min(ARM_MAX, Math.max(0, st.hold - ARM_CHARGE) * ARM_SPEED)
    if (ext > 0 && !grabs.has(arm.id)) {
      grabs.set(arm.id, { attach: { kind: 'arm', arm }, x: 0, y: 0, sigma: out.fore * 0.4 })
      shout = { text: 'GUM GUM...', t: performance.now() }
    }
    const g = grabs.get(arm.id)
    if (g) {
      const [hx, hy] = st.hand
      g.x = hx + (out.dx * ext) / aspect
      g.y = hy + out.dy * ext
    }
  })
}

// ---------- physics ----------

let ribbons = [] // { onFace, a, b, srcA, srcB, w, srcW, taper, aHome, bHome, erase } in screen coords
const snaps = [] // ribbons you let go of, shooting back

function offsetAt(x, y) {
  const c = Math.max(0, Math.min(COLS, Math.round(x * COLS)))
  const r = Math.max(0, Math.min(ROWS, Math.round(y * ROWS)))
  const i = (r * (COLS + 1) + c) * 2
  return [offset[i], offset[i + 1]]
}

function armRibbon(g) {
  // your forearm and hand, stretched out to where the hand has shot to
  if (!pose) return null
  const arm = g.attach.arm
  const E = pose[arm.elbow]
  const W = pose[arm.wrist]
  const I = pose[arm.index]
  const [hx, hy] = handPoint(arm)
  const w = Math.hypot((W.x - E.x) * aspect, W.y - E.y) * 0.45
  return {
    a: [E.x, E.y],
    b: [I.x + g.x - hx, I.y + g.y - hy],
    srcA: [E.x, E.y],
    srcB: [I.x, I.y],
    w,
    srcW: w,
    taper: 0.3,
    aHome: [E.x, E.y],
    bHome: [I.x, I.y],
    erase: { from: [E.x, E.y], to: [I.x, I.y], w: w * 1.6 },
  }
}

function step() {
  const active = []
  ribbons = []

  for (const g of grabs.values()) {
    if (g.attach.kind === 'finger') {
      const a = g.attach
      updateFinger(a, [g.x * aspect + a.pinchOffset[0], g.y + a.pinchOffset[1]])
      continue
    }
    if (g.attach.kind === 'arm') {
      g.ribbon = armRibbon(g)
      if (g.ribbon) ribbons.push(g.ribbon)
      continue
    }
    const [ax, ay] = anchor(g)
    const dx = g.x - ax
    const dy = g.y - ay
    // only tug the face a little, the ribbon does the rest
    const len = Math.hypot(dx * aspect, dy)
    const cap = Math.min(1, (g.sigma * TUG) / (len || 1))
    active.push({ ax, ay, dx: dx * cap, dy: dy * cap, s2: 2 * g.sigma * g.sigma })
    g.ribbon = {
      onFace: true,
      a: [ax, ay],
      b: [g.x, g.y],
      srcA: [ax, ay],
      srcB: [ax, ay],
      w: g.sigma * 2.6,
      srcW: g.sigma * 0.8,
      aHome: [ax, ay],
      bHome: [ax, ay],
    }
    ribbons.push(g.ribbon)
  }

  const rate = active.length ? FOLLOW : SNAP
  let biggest = 0

  for (let i = 0; i < vertCount * 2; i += 2) {
    let tx = 0
    let ty = 0
    for (const a of active) {
      const ddx = (rest[i] - a.ax) * aspect
      const ddy = rest[i + 1] - a.ay
      const w = Math.exp(-(ddx * ddx + ddy * ddy) / a.s2)
      tx += a.dx * w
      ty += a.dy * w
    }
    offset[i] += (tx - offset[i]) * rate
    offset[i + 1] += (ty - offset[i + 1]) * rate
    biggest = Math.max(biggest, Math.abs(offset[i]), Math.abs(offset[i + 1]))
  }

  // once it's basically back, reset exactly so no leftover stretch hangs around
  if (!active.length && biggest < 0.002) offset.fill(0)

  // face ribbons start where the tugged skin is now, so they join up with your face
  for (const r of ribbons) {
    if (!r.onFace) continue
    const [ox, oy] = offsetAt(r.a[0], r.a[1])
    r.a = [r.a[0] + ox, r.a[1] + oy]
  }

  // let go ribbons shoot back into you, then disappear
  for (let k = snaps.length - 1; k >= 0; k--) {
    const r = snaps[k]
    for (const [p, home] of [[r.a, r.aHome], [r.b, r.bHome]]) {
      p[0] += (home[0] - p[0]) * RIBBON_SNAP
      p[1] += (home[1] - p[1]) * RIBBON_SNAP
    }
    if (Math.hypot((r.b[0] - r.bHome[0]) * aspect, r.b[1] - r.bHome[1]) < 0.004) snaps.splice(k, 1)
  }

  // let go fingers spring back onto the real finger, then the real one takes over
  const now = performance.now()
  for (let k = fingerSnaps.length - 1; k >= 0; k--) {
    const sn = fingerSnaps[k]
    const t = (now - sn.start) / 1000 / FINGER_SNAP_TIME
    if (t >= 1) {
      freeFinger(sn.a.finger)
      fingerSnaps.splice(k, 1)
      continue
    }
    trackFinger(sn.a, -1)
    const home = fingerHome(sn.a)
    const e = easeSpring(t)
    updateFinger(sn.a, [sn.from[0] + (home[0] - sn.from[0]) * e, sn.from[1] + (home[1] - sn.from[1]) * e])
  }
}

// across the ribbon: where each column sits (-1 to 1 edge to edge) and how bright it is.
// dark edges and a soft shine just off center make it read as a round tube of skin
const RIBBON_COLS = [
  [-1, 0.5],
  [-0.4, 0.95],
  [0.25, 1.15],
  [0.6, 0.95],
  [1, 0.5],
]

function ribbonVerts(r, out) {
  const dx = (r.b[0] - r.a[0]) * aspect
  const dy = r.b[1] - r.a[1]
  const len = Math.hypot(dx, dy)
  if (len < 0.01) return

  // how far past its normal length it is. like the finger, the rubber effects (curve,
  // thinning, taper, shading) all scale with this, so as it comes back it turns back
  // into exactly the skin it came from instead of popping
  const sdx = (r.srcB[0] - r.srcA[0]) * aspect
  const sdy = r.srcB[1] - r.srcA[1]
  const slen = Math.hypot(sdx, sdy)
  const stretch = Math.max(0, len - slen)
  const amt = Math.min(1, stretch / (2 * r.w))

  // bend it into a curve that sags downward like heavy rubber
  let nx = -dy / len
  let ny = dx / len
  if (ny < 0) {
    nx = -nx
    ny = -ny
  }
  const bow = Math.min(stretch * 0.12, 0.08)
  const mx = (r.a[0] + r.b[0]) / 2 + (nx * bow) / aspect
  const my = (r.a[1] + r.b[1]) / 2 + ny * bow
  const curve = (t) => {
    const u = 1 - t
    return [u * u * r.a[0] + 2 * u * t * mx + t * t * r.b[0], u * u * r.a[1] + 2 * u * t * my + t * t * r.b[1]]
  }

  // across the source skin. a face grab samples one slice of skin, so it streaks like rubber
  const spx = slen > 0.001 ? -sdy / slen : -dy / len
  const spy = slen > 0.001 ? sdx / slen : dx / len

  // the longer it gets the thinner it gets, like real rubber
  const thin = 1 / (1 + stretch)

  const rows = []
  for (let j = 0; j <= RIBBON_SEGS; j++) {
    const t = j / RIBBON_SEGS
    const [cx, cy] = curve(t)
    // the curve's direction here, so the width is always square to it
    const [ax, ay] = curve(Math.max(0, t - 0.01))
    const [bx, by] = curve(Math.min(1, t + 0.01))
    const tx = (bx - ax) * aspect
    const ty = by - ay
    const tl = Math.hypot(tx, ty) || 1
    const px = -ty / tl
    const py = tx / tl
    // tapers toward your fingers and pinches in a bit in the middle
    const half = 0.5 * r.w * thin * (1 - (r.taper ?? 0.55) * amt * t) * (1 - 0.15 * amt * Math.sin(Math.PI * t))
    const sx = r.srcA[0] + (r.srcB[0] - r.srcA[0]) * t
    const sy = r.srcA[1] + (r.srcB[1] - r.srcA[1]) * t
    const sHalf = 0.5 * r.srcW
    rows.push(
      RIBBON_COLS.map(([side, shade]) => [
        cx + (px * half * side) / aspect,
        cy + py * half * side,
        sx + (spx * sHalf * side) / aspect,
        sy + spy * sHalf * side,
        1 + (shade - 1) * amt,
      ]),
    )
  }
  for (let j = 0; j < RIBBON_SEGS; j++) {
    for (let c = 0; c < RIBBON_COLS.length - 1; c++) {
      const a = rows[j][c]
      const b = rows[j][c + 1]
      const d = rows[j + 1][c]
      const e = rows[j + 1][c + 1]
      for (const v of [a, b, d, b, e, d]) out.push(...v)
    }
  }
}

function eraseVerts(e, out) {
  // a fat line over where the finger / arm really is, in screen coords
  const dx = (e.to[0] - e.from[0]) * aspect
  const dy = e.to[1] - e.from[1]
  const len = Math.hypot(dx, dy)
  if (len < 0.001) return
  const half = e.w / 2
  const px = dy / len / aspect
  const py = -dx / len
  // a bit past both ends so the knuckle and fingertip are covered too
  const ext = half / len
  const t0 = -ext
  const t1 = 1 + ext
  const at = (t, side) => [
    e.from[0] + (e.to[0] - e.from[0]) * t + px * half * side,
    e.from[1] + (e.to[1] - e.from[1]) * t + py * half * side,
  ]
  const [a, b, c, d] = [at(t0, 1), at(t0, -1), at(t1, 1), at(t1, -1)]
  out.push(...a, ...b, ...c, ...b, ...d, ...c)
}

function drawErase() {
  const hole = []
  for (const r of [...ribbons, ...snaps]) if (r.erase) eraseVerts(r.erase, hole)
  for (const a of activeFingers()) for (const e of fingerHoles(a)) eraseVerts(e, hole)
  if (!hole.length) return
  if (!pyramid || pyramid.aspect !== aspect) buildPyramid()

  gl.disable(gl.DEPTH_TEST)
  const top = pyramid.length - 1

  // the camera image at the biggest size, with the hole cut out
  bindTex(0, tex)
  fillPass('copy', screenQuad, pyramid[0].push, pyramid[0], (u) => gl.uniform1i(u.src, 0))
  fillPass('hole', hole, pyramid[0].push, pyramid[0])

  // shrink it down, ignoring the hole
  for (let i = 1; i <= top; i++) {
    const from = pyramid[i - 1]
    bindTex(0, from.push.tex)
    fillPass('down', screenQuad, pyramid[i].push, pyramid[i], (u) => {
      gl.uniform1i(u.src, 0)
      gl.uniform2f(u.srcSize, from.w, from.h)
    })
  }

  // grow it back up, filling the hole from the smaller copy each time
  bindTex(0, pyramid[top].push.tex)
  fillPass('top', screenQuad, pyramid[top].filled, pyramid[top], (u) => {
    gl.uniform1i(u.src, 0)
    gl.uniform2f(u.size, pyramid[top].w, pyramid[top].h)
  })
  for (let i = top - 1; i >= 0; i--) {
    const lvl = pyramid[i]
    bindTex(0, lvl.push.tex)
    bindTex(1, pyramid[i + 1].filled.tex)
    fillPass('up', screenQuad, lvl.filled, lvl, (u) => {
      gl.uniform1i(u.src, 0)
      gl.uniform1i(u.coarse, 1)
      gl.uniform2f(u.size, lvl.w, lvl.h)
    })
  }

  // paint the filled image over the hole on screen
  bindTex(0, pyramid[0].filled.tex)
  fillPass('paint', hole, null, null, (u) => gl.uniform1i(u.src, 0))

  // put things back how the rest of the drawing expects them
  bindTex(1, null)
  bindTex(0, tex)
  gl.enable(gl.DEPTH_TEST)
}

function drawRibbons() {
  const out = []
  for (const r of [...ribbons, ...snaps]) ribbonVerts(r, out)
  if (!out.length) return

  gl.useProgram(ribbonProg)
  gl.bindBuffer(gl.ARRAY_BUFFER, ribbonBuf)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(out), gl.STREAM_DRAW)
  const [pos, src, shade] = ribbonLocs
  for (const loc of ribbonLocs) gl.enableVertexAttribArray(loc)
  gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 20, 0)
  gl.vertexAttribPointer(src, 2, gl.FLOAT, false, 20, 8)
  gl.vertexAttribPointer(shade, 1, gl.FLOAT, false, 20, 16)
  gl.disable(gl.DEPTH_TEST)
  gl.drawArrays(gl.TRIANGLES, 0, out.length / 5)
  gl.enable(gl.DEPTH_TEST)
  for (const loc of ribbonLocs) gl.disableVertexAttribArray(loc)
}

function drawMesh() {
  gl.useProgram(prog)
  gl.enableVertexAttribArray(restLoc)
  gl.bindBuffer(gl.ARRAY_BUFFER, restBuf)
  gl.vertexAttribPointer(restLoc, 2, gl.FLOAT, false, 0, 0)
  gl.enableVertexAttribArray(offsetLoc)
  gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuf)
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, offset)
  gl.vertexAttribPointer(offsetLoc, 2, gl.FLOAT, false, 0, 0)
  gl.clear(gl.DEPTH_BUFFER_BIT)
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)
}

// ---------- finger stretch ----------

// the grabbed finger's real picture is cut out (trimmed to skin colour), turned into a
// mesh, and bent with ARAP (see arap.js) from its knuckle to your pinch. the further it
// stretches the thinner and shinier it gets, and at normal length it's exactly the real
// finger again, so it springs back without a seam

const FINGER_ROWS = 14 // mesh rows along the finger
const FINGER_COLS = 5 // and across it
const FINGER_THIN = 0.15 // how much it narrows when stretched
const FINGER_SHINE = 0.2 // how much it shines when stretched (capped at 0.25)
const FINGER_SNAP_TIME = 0.3 // seconds to spring back
const SKIN_TOLERANCE = 22 // how close in colour (YCrCb) a pixel must be to count as finger
const MASK_WIDTH = 640 // the skin cut-out is worked out at this size

const fingerSnaps = [] // { a, from, start }

const fingerVert = `
attribute vec2 pos;
attribute vec2 src;
attribute float across;
varying vec2 uv;
varying float vAcross;
void main() {
  uv = vec2(1.0 - src.x, src.y);
  vAcross = across;
  gl_Position = vec4(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
}`

const fingerFrag = `
precision mediump float;
uniform sampler2D snap;
uniform sampler2D mask;
uniform float shine;
varying vec2 uv;
varying float vAcross;
void main() {
  float a = texture2D(mask, uv).r;
  if (a < 0.02) discard;
  vec3 c = texture2D(snap, uv).rgb;
  // a soft shiny band a little off center, brightening the skin without going white
  float band = exp(-pow((vAcross - 0.25) / 0.22, 2.0));
  gl_FragColor = vec4(min(c * (1.0 + shine * band), vec3(1.0)), a);
}`

const fingerProg = gl.createProgram()
gl.attachShader(fingerProg, compile(gl.VERTEX_SHADER, fingerVert))
gl.attachShader(fingerProg, compile(gl.FRAGMENT_SHADER, fingerFrag))
gl.linkProgram(fingerProg)
const fingerLocs = ['pos', 'src', 'across'].map((n) => gl.getAttribLocation(fingerProg, n))
const fingerU = {
  snap: gl.getUniformLocation(fingerProg, 'snap'),
  mask: gl.getUniformLocation(fingerProg, 'mask'),
  shine: gl.getUniformLocation(fingerProg, 'shine'),
}
const fingerBuf = gl.createBuffer()

const maskCanvas = document.createElement('canvas')
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })

// points are kept in "iso" space: screen x times aspect, so distances are the same both ways
const iso = (p) => [p.x * aspect, p.y]

function chainPoints(a) {
  return a.chain.map((i) => iso(a.hand[i]))
}

function distToChain(x, y, P) {
  let best = Infinity
  for (let i = 0; i < P.length - 1; i++) {
    const [ax, ay] = P[i]
    const bx = P[i + 1][0] - ax
    const by = P[i + 1][1] - ay
    const t = Math.max(0, Math.min(1, ((x - ax) * bx + (y - ay) * by) / (bx * bx + by * by || 1)))
    best = Math.min(best, Math.hypot(x - ax - bx * t, y - ay - by * t))
  }
  return best
}

function makeTexture() {
  const t = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, t)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  return t
}

function skinMask(P, r) {
  // which pixels near the finger are actually finger: inside the landmark capsule, and
  // the same skin colour as the middle of the finger (so the wall beside it is left out)
  const W = MASK_WIDTH
  const H = Math.round(MASK_WIDTH / aspect)
  maskCanvas.width = W
  maskCanvas.height = H
  maskCtx.drawImage(video, 0, 0, W, H)
  const out = new Uint8Array(W * H)

  // the area around the finger, in mask pixels (the video isn't mirrored, the screen is)
  const xs = P.map((p) => p[0] / aspect)
  const ys = P.map((p) => p[1])
  const pad = r / aspect + 2 / W
  const x0 = Math.max(0, Math.floor((1 - Math.max(...xs) - pad) * W))
  const x1 = Math.min(W, Math.ceil((1 - Math.min(...xs) + pad) * W))
  const y0 = Math.max(0, Math.floor((Math.min(...ys) - r - 2 / H) * H))
  const y1 = Math.min(H, Math.ceil((Math.max(...ys) + r + 2 / H) * H))
  if (x1 <= x0 || y1 <= y0) return { out, W, H }
  const px = maskCtx.getImageData(x0, y0, x1 - x0, y1 - y0).data

  const inside = []
  const crs = []
  const cbs = []
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const d = distToChain((1 - (x + 0.5) / W) * aspect, (y + 0.5) / H, P)
      if (d >= r) continue
      const i = ((y - y0) * (x1 - x0) + (x - x0)) * 4
      const R = px[i]
      const G = px[i + 1]
      const B = px[i + 2]
      const Y = 0.299 * R + 0.587 * G + 0.114 * B
      const cr = (R - Y) * 0.713 + 128
      const cb = (B - Y) * 0.564 + 128
      inside.push([y * W + x, cr, cb])
      if (d < r * 0.5) {
        crs.push(cr)
        cbs.push(cb)
      }
    }
  }
  const median = (v) => v.sort((a, b) => a - b)[v.length >> 1] ?? 128
  const cr0 = median(crs)
  const cb0 = median(cbs)
  let kept = 0
  for (const [i, cr, cb] of inside) {
    if (Math.abs(cr - cr0) <= SKIN_TOLERANCE && Math.abs(cb - cb0) <= SKIN_TOLERANCE) {
      out[i] = 255
      kept++
    }
  }
  // if the colour check threw most of it away it probably went wrong, so just use the capsule
  if (kept < inside.length * 0.22) for (const [i] of inside) out[i] = 255
  return { out, W, H }
}

function buildFinger(a) {
  const P = chainPoints(a)
  const r = a.width * 0.95 * 0.5
  const segs = P.slice(1).map((p, i) => Math.hypot(p[0] - P[i][0], p[1] - P[i][1]))
  const fingerLen = segs.reduce((s, l) => s + l, 0)
  if (fingerLen < 0.01) return null
  const total = fingerLen + r * 0.8 // a little past the fingertip for the rounded end

  // a point along the finger, carrying on straight past the tip
  const along = (d) => {
    for (let i = 0; i < segs.length; i++) {
      if (d <= segs[i] || i === segs.length - 1) {
        const t = d / (segs[i] || 1)
        return [P[i][0] + (P[i + 1][0] - P[i][0]) * t, P[i][1] + (P[i + 1][1] - P[i][1]) * t]
      }
      d -= segs[i]
    }
  }

  const rest = []
  const across = []
  const half = r * 1.3 // wider than the finger, the skin cut-out decides the real edge
  for (let i = 0; i < FINGER_ROWS; i++) {
    const d = (i / (FINGER_ROWS - 1)) * total
    const [cx, cy] = along(d)
    const [ax, ay] = along(Math.max(0, d - total * 0.05))
    const [bx, by] = along(Math.min(total, d + total * 0.05))
    const l = Math.hypot(bx - ax, by - ay) || 1
    const nx = -(by - ay) / l
    const ny = (bx - ax) / l
    for (let c = 0; c < FINGER_COLS; c++) {
      const side = -1 + (2 * c) / (FINGER_COLS - 1)
      rest.push([cx + nx * half * side, cy + ny * half * side])
      across.push(side)
    }
  }
  const tris = []
  for (let i = 0; i < FINGER_ROWS - 1; i++) {
    for (let c = 0; c < FINGER_COLS - 1; c++) {
      const v = i * FINGER_COLS + c
      tris.push([v, v + 1, v + FINGER_COLS], [v + 1, v + FINGER_COLS + 1, v + FINGER_COLS])
    }
  }
  // the knuckle end stays pinned to your hand, the fingertip end is what you pull
  const anchors = [...Array(2 * FINGER_COLS).keys()]
  const tipV = (FINGER_ROWS - 1) * FINGER_COLS + (FINGER_COLS >> 1)

  const snap = makeTexture()
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video)
  const m = skinMask(P, r)
  const mask = makeTexture()
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, m.W, m.H, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, m.out)
  gl.bindTexture(gl.TEXTURE_2D, tex)

  return {
    rest,
    across,
    tris,
    anchors,
    tipV,
    solver: new ARAP(rest, tris, [...anchors, tipV]),
    knuckle0: P[0],
    natural: Math.hypot(rest[tipV][0] - P[0][0], rest[tipV][1] - P[0][1]),
    snap,
    mask,
    def: null,
    target: null,
    stretch: 0,
    shine: 0,
  }
}

function freeFinger(F) {
  gl.deleteTexture(F.snap)
  gl.deleteTexture(F.mask)
}

// where the fingertip end of the mesh would be with the finger at rest, following your hand
function fingerHome(a) {
  const F = a.finger
  const [kx, ky] = iso(a.hand[a.mcpIdx])
  return [F.rest[F.tipV][0] + kx - F.knuckle0[0], F.rest[F.tipV][1] + ky - F.knuckle0[1]]
}

function updateFinger(a, target) {
  const F = a.finger
  const [kx, ky] = iso(a.hand[a.mcpIdx])
  const dx = kx - F.knuckle0[0]
  const dy = ky - F.knuckle0[1]
  const targets = new Map(F.anchors.map((v) => [v, [F.rest[v][0] + dx, F.rest[v][1] + dy]]))
  targets.set(F.tipV, target)
  const def = F.solver.solve(targets)

  // how far past its normal length it is (0 = normal). drives the thinning and shine,
  // so both fade away by themselves as it springs back
  const amt = Math.max(0, Math.min(2, Math.hypot(target[0] - kx, target[1] - ky) / F.natural - 1))
  const thin = FINGER_THIN * amt
  if (thin > 0) {
    const ax = target[0] - kx
    const ay = target[1] - ky
    const L2 = ax * ax + ay * ay || 1
    const pinned = new Set([...F.anchors, F.tipV])
    def.forEach((p, i) => {
      if (pinned.has(i)) return
      // pull it toward the knuckle -> pinch line
      const t = ((p[0] - kx) * ax + (p[1] - ky) * ay) / L2
      p[0] += (kx + ax * t - p[0]) * thin
      p[1] += (ky + ay * t - p[1]) * thin
    })
  }
  F.def = def
  F.target = target
  F.shine = Math.min(0.25, FINGER_SHINE * amt)
  // how far the tip is from where the real one is, for fading the fill in and out
  const [hx, hy] = fingerHome(a)
  F.stretch = Math.hypot(target[0] - hx, target[1] - hy) / F.natural
}

function activeFingers() {
  const list = []
  for (const g of grabs.values()) if (g.attach.kind === 'finger' && g.attach.finger.def) list.push(g.attach)
  for (const sn of fingerSnaps) list.push(sn.a)
  return list
}

function fingerHoles(a) {
  // cover the real finger with background, but only once it's actually moved away,
  // so there's no blurry halo when it's back in place
  const fade = Math.min(1, a.finger.stretch * 5)
  if (fade <= 0) return []
  const P = a.chain.map((i) => [a.hand[i].x, a.hand[i].y])
  return P.slice(1).map((p, i) => ({ from: P[i], to: p, w: a.width * 1.3 * fade }))
}

function drawFingers() {
  const list = activeFingers()
  if (!list.length) return
  gl.useProgram(fingerProg)
  gl.disable(gl.DEPTH_TEST)
  gl.enable(gl.BLEND)
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
  for (const loc of fingerLocs) gl.enableVertexAttribArray(loc)
  gl.uniform1i(fingerU.snap, 0)
  gl.uniform1i(fingerU.mask, 1)

  for (const a of list) {
    const F = a.finger
    const out = []
    for (const t of F.tris) {
      for (const v of t) {
        out.push(F.def[v][0] / aspect, F.def[v][1], F.rest[v][0] / aspect, F.rest[v][1], F.across[v])
      }
    }
    bindTex(0, F.snap)
    bindTex(1, F.mask)
    gl.uniform1f(fingerU.shine, F.shine)
    gl.bindBuffer(gl.ARRAY_BUFFER, fingerBuf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(out), gl.STREAM_DRAW)
    const [pos, src, across] = fingerLocs
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 20, 0)
    gl.vertexAttribPointer(src, 2, gl.FLOAT, false, 20, 8)
    gl.vertexAttribPointer(across, 1, gl.FLOAT, false, 20, 16)
    gl.drawArrays(gl.TRIANGLES, 0, out.length / 5)
  }

  for (const loc of fingerLocs) gl.disableVertexAttribArray(loc)
  gl.disable(gl.BLEND)
  gl.enable(gl.DEPTH_TEST)
  bindTex(1, null)
  bindTex(0, tex)
}

// ---------- drawing ----------

function drawHud() {
  const w = hud.width
  const h = hud.height
  ctx.clearRect(0, 0, w, h)

  for (const p of pinchPoints) {
    ctx.beginPath()
    ctx.arc(p.x * w, p.y * h, p.isPinch ? 10 : 6, 0, Math.PI * 2)
    ctx.fillStyle = p.isPinch ? '#e63946' : 'rgba(255,255,255,0.6)'
    ctx.fill()
  }

  // charge ring around the hand while you hold your arm out
  for (const st of armState) {
    if (!st.hand || st.hold < 0.15 || st.hold > ARM_CHARGE) continue
    ctx.beginPath()
    ctx.arc(st.hand[0] * w, st.hand[1] * h, h * 0.05, -Math.PI / 2, -Math.PI / 2 + (st.hold / ARM_CHARGE) * Math.PI * 2)
    ctx.lineWidth = h * 0.012
    ctx.strokeStyle = '#f4c430'
    ctx.stroke()
  }

  const age = performance.now() - shout.t
  if (shout.text && age < 900) {
    const scale = 1 + (age / 900) * 0.4
    ctx.save()
    ctx.globalAlpha = 1 - age / 900
    ctx.translate(w / 2, h * 0.15)
    ctx.scale(scale, scale)
    ctx.font = `900 ${Math.round(h * 0.09)}px system-ui, sans-serif`
    ctx.textAlign = 'center'
    ctx.lineWidth = h * 0.012
    ctx.strokeStyle = '#111'
    ctx.strokeText(shout.text, 0, 0)
    ctx.fillStyle = '#f4c430'
    ctx.fillText(shout.text, 0, 0)
    ctx.restore()
  }

  if (faceLandmarker && !face) {
    ctx.font = `${Math.round(h * 0.035)}px system-ui, sans-serif`
    ctx.textAlign = 'center'
    ctx.fillStyle = 'rgba(255,255,255,0.8)'
    ctx.fillText("can't see your face", w / 2, h * 0.95)
  }
}

let lastVideoTime = -1
let lastLoop = performance.now()

function loop() {
  const now = performance.now()
  const dt = Math.min(0.1, (now - lastLoop) / 1000)
  lastLoop = now

  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime
    if (faceLandmarker) {
      const res = faceLandmarker.detectForVideo(video, now)
      face = res.faceLandmarks[0]?.map((p) => ({ x: 1 - p.x, y: p.y })) ?? null
    }
    if (poseLandmarker) {
      const res = poseLandmarker.detectForVideo(video, now)
      pose = res.landmarks[0]?.map((p) => ({ x: 1 - p.x, y: p.y, visibility: p.visibility ?? 1 })) ?? null
    }
    if (handLandmarker) updateHands(handLandmarker.detectForVideo(video, now))
  }

  updateArms(dt)
  step()

  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video)
  drawMesh()
  drawErase()
  drawRibbons()
  drawFingers()

  drawHud()
  requestAnimationFrame(loop)
}

// ---------- start ----------

async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(WASM)
  ;[faceLandmarker, handLandmarker, poseLandmarker, segmenter] = await Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 2,
    }),
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    }),
    ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: SEG_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    }),
  ])
}

document.getElementById('go').addEventListener('click', async () => {
  statusText.textContent = 'starting camera...'
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: 'user' },
      audio: false,
    })
    await video.play()
  } catch (err) {
    statusText.textContent = 'no camera :( ' + err.message
    return
  }

  statusText.textContent = 'loading face, hand and body models...'
  try {
    await loadModels()
  } catch (err) {
    // still playable with the mouse, just without tracking
    console.warn('models failed to load', err)
    faceLandmarker = handLandmarker = poseLandmarker = segmenter = null
  }

  startScreen.remove()
  resize()
  requestAnimationFrame(loop)
})
