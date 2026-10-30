import {
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
  ImageSegmenter,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'

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

// how much of the way to where it should be the skin moves each frame. no springs, so
// it slides smoothly back to normal when you let go instead of bouncing around
const FOLLOW = 0.25 // while you're pulling
const SETTLE = 0.2 // after you let go

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
uniform sampler2D mask;
uniform float useMask;
varying vec2 uv;
void main() {
  // the arm layer only draws you, never the background, so your curtains stay put
  if (useMask > 0.5 && texture2D(mask, uv).r < 0.5) discard;
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
// arm stretches get their own layer, drawn on top with the background cut out
const armOffset = new Float32Array(vertCount * 2)

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
const restLoc = gl.getAttribLocation(prog, 'rest')
const offsetBuf = attrib('offset', offset, gl.DYNAMIC_DRAW)
const armOffsetBuf = attrib('offset', armOffset, gl.DYNAMIC_DRAW)
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

// which pixels are you (255) and which are background (0), from the segmenter.
// starts as "all you" so it still works if the models didn't load
const maskTex = gl.createTexture()
gl.activeTexture(gl.TEXTURE1)
gl.bindTexture(gl.TEXTURE_2D, maskTex)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([255]))
gl.activeTexture(gl.TEXTURE0)
gl.uniform1i(gl.getUniformLocation(prog, 'tex'), 0)
gl.uniform1i(gl.getUniformLocation(prog, 'mask'), 1)
const useMaskLoc = gl.getUniformLocation(prog, 'useMask')

let personMask = null

function updatePersonMask() {
  const res = segmenter.segmentForVideo(video, performance.now())
  const m = res.categoryMask
  const cats = m.getAsUint8Array()
  if (!personMask || personMask.length !== cats.length) personMask = new Uint8Array(cats.length)
  for (let i = 0; i < cats.length; i++) personMask[i] = cats[i] === 0 ? 0 : 255 // 0 is background
  const w = m.width
  const h = m.height
  res.close()
  gl.activeTexture(gl.TEXTURE1)
  gl.bindTexture(gl.TEXTURE_2D, maskTex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, w, h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, personMask)
  gl.activeTexture(gl.TEXTURE0)
}

// ---------- erase (fill) ----------

// paints over where your real arm and hand are while the arm is stretched, so you only
// see the stretched one.
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
  if (!grabs.has(id)) return
  grabs.delete(id)
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
let pinchPoints = []
let handPoints = [] // every hand's 21 points in screen space, for the H guides

// press H to show what the camera thinks your hands are doing
let showHands = false
addEventListener('keydown', (e) => {
  if (e.key === 'h' || e.key === 'H') showHands = !showHands
})

function updateHands(result) {
  pinchPoints = []
  handPoints = result.landmarks.map((hand) => hand.map((p) => ({ x: 1 - p.x, y: p.y })))
  const seen = [false, false]

  result.landmarks.forEach((hand, h) => {
    if (h > 1) return
    seen[h] = true
    const thumb = hand[4]
    const index = hand[8]
    const size = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y)
    const gap = Math.hypot(thumb.x - index.x, thumb.y - index.y)
    const x = 1 - (thumb.x + index.x) / 2
    const y = (thumb.y + index.y) / 2

    // a bit of hysteresis so it doesn't flicker between pinch and not pinch
    const isPinch = pinching[h] ? gap < size * 0.45 : gap < size * 0.3
    pinchPoints.push({ x, y, isPinch })
    const wrist = { x: 1 - hand[0].x, y: hand[0].y }

    if (isPinch && !pinching[h]) startGrab('h' + h, x, y, wrist)
    else if (isPinch) moveGrab('h' + h, x, y)
    else if (pinching[h]) endGrab('h' + h)
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

function targets(list, i) {
  let tx = 0
  let ty = 0
  for (const a of list) {
    let cx = a.ax
    let cy = a.ay
    let ramp = 1
    if (a.root) {
      // closest point on the shoulder -> hand line, and how far along it we are
      const px = (rest[i] - a.root.x) * aspect
      const py = rest[i + 1] - a.root.y
      ramp = Math.max(0, Math.min(1, (px * a.rx + py * a.ry) / a.rlen2))
      cx = a.root.x + (a.ax - a.root.x) * ramp
      cy = a.root.y + (a.ay - a.root.y) * ramp
    }
    const ddx = (rest[i] - cx) * aspect
    const ddy = rest[i + 1] - cy
    const w = ramp * Math.exp(-(ddx * ddx + ddy * ddy) / a.s2)
    tx += a.dx * w
    ty += a.dy * w
  }
  return [tx, ty]
}

let armMoving = false
let armAmount = 0 // how far the arm layer is stretched, in screen heights
let lastArm = null // the arm being stretched (kept while it slides back)

function step() {
  const pulls = []
  const arms = []
  for (const g of grabs.values()) {
    const [ax, ay] = anchor(g)
    const dx = g.x - ax
    const dy = g.y - ay
    // arm stretches ramp up from the shoulder so the whole arm stretches, not just the hand
    const root = g.attach.kind === 'arm' && pose ? pose[g.attach.arm.shoulder] : null
    const rx = root ? (ax - root.x) * aspect : 0
    const ry = root ? ay - root.y : 0
    const a = { ax, ay, dx, dy, s2: 2 * g.sigma * g.sigma, root, rx, ry, rlen2: rx * rx + ry * ry || 1 }
    ;(g.attach.kind === 'arm' ? arms : pulls).push(a)
    if (g.attach.kind === 'arm') lastArm = g.attach.arm
  }

  let biggest = 0
  for (let i = 0; i < vertCount * 2; i += 2) {
    const [tx, ty] = targets(pulls, i)
    const rate = pulls.length ? FOLLOW : SETTLE
    offset[i] += (tx - offset[i]) * rate
    offset[i + 1] += (ty - offset[i + 1]) * rate

    const [ax, ay] = targets(arms, i)
    const armRate = arms.length ? FOLLOW : SETTLE
    armOffset[i] += (ax - armOffset[i]) * armRate
    armOffset[i + 1] += (ay - armOffset[i + 1]) * armRate
    biggest = Math.max(biggest, Math.abs(armOffset[i]), Math.abs(armOffset[i + 1]))
  }
  armMoving = arms.length > 0 || biggest > 0.001
  armAmount = biggest
}

// ---------- filling in the real arm ----------

function armHoles() {
  // your real forearm and hand, as fat lines, fading in as the arm stretches so there's
  // no blurry patch when it's back to normal
  if (!armMoving || !lastArm || !pose) return []
  const fade = Math.min(1, armAmount / 0.03)
  if (fade <= 0) return []
  const E = pose[lastArm.elbow]
  const W = pose[lastArm.wrist]
  const I = pose[lastArm.index]
  const fore = Math.hypot((W.x - E.x) * aspect, W.y - E.y)
  // carry on past the index knuckle to cover your fingers too
  const tip = { x: I.x + (I.x - W.x) * 1.2, y: I.y + (I.y - W.y) * 1.2 }
  return [
    { from: [E.x, E.y], to: [W.x, W.y], w: fore * 0.55 * fade },
    { from: [W.x, W.y], to: [tip.x, tip.y], w: fore * 0.9 * fade },
  ]
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
  for (const e of armHoles()) eraseVerts(e, hole)
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

// ---------- drawing ----------

// how the 21 hand points join up, for the H guides
const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]
const FINGERTIPS = [4, 8, 12, 16, 20]

function drawHud() {
  const w = hud.width
  const h = hud.height
  ctx.clearRect(0, 0, w, h)

  if (showHands) {
    for (const hand of handPoints) {
      ctx.lineWidth = Math.max(2, h * 0.004)
      ctx.strokeStyle = 'rgba(46, 204, 113, 0.9)'
      for (const [a, b] of HAND_BONES) {
        ctx.beginPath()
        ctx.moveTo(hand[a].x * w, hand[a].y * h)
        ctx.lineTo(hand[b].x * w, hand[b].y * h)
        ctx.stroke()
      }
      hand.forEach((p, i) => {
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, Math.max(3, h * 0.006), 0, Math.PI * 2)
        ctx.fillStyle = FINGERTIPS.includes(i) ? '#e63946' : '#f4c430'
        ctx.fill()
      })
    }
  }

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
  // only work out which pixels are you while an arm is stretching, it isn't free
  if (segmenter && armMoving) updatePersonMask()

  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video)

  // the whole picture, with face / body pulls
  gl.useProgram(prog)
  gl.enableVertexAttribArray(restLoc)
  gl.bindBuffer(gl.ARRAY_BUFFER, restBuf)
  gl.vertexAttribPointer(restLoc, 2, gl.FLOAT, false, 0, 0)
  gl.enableVertexAttribArray(offsetLoc)
  gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuf)
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, offset)
  gl.vertexAttribPointer(offsetLoc, 2, gl.FLOAT, false, 0, 0)
  gl.uniform1f(useMaskLoc, 0)
  gl.clear(gl.DEPTH_BUFFER_BIT)
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)

  // then a stretched arm on top, just you with the background cut out, so the
  // background underneath never moves
  if (armMoving) {
    drawErase()
    // the fill used its own shaders, so set the mesh back up
    gl.useProgram(prog)
    gl.enableVertexAttribArray(restLoc)
    gl.bindBuffer(gl.ARRAY_BUFFER, restBuf)
    gl.vertexAttribPointer(restLoc, 2, gl.FLOAT, false, 0, 0)
    bindTex(1, maskTex)
    gl.activeTexture(gl.TEXTURE0)
    gl.enableVertexAttribArray(offsetLoc)
    gl.bindBuffer(gl.ARRAY_BUFFER, armOffsetBuf)
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, armOffset)
    gl.vertexAttribPointer(offsetLoc, 2, gl.FLOAT, false, 0, 0)
    gl.uniform1f(useMaskLoc, 1)
    gl.clear(gl.DEPTH_BUFFER_BIT)
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)
  }

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
