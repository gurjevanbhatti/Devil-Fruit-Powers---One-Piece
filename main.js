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

// spring tuning. lower damping = more wobble when you let go
const STIFFNESS = 0.16
const DAMPING = 0.84

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
let maxStretch = 0
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
const vel = new Float32Array(vertCount * 2)

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

attrib('rest', rest, gl.STATIC_DRAW)
const offsetBuf = attrib('offset', offset, gl.DYNAMIC_DRAW)

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
  if (grabs.size === 0) {
    if (maxStretch > 0.35) shout = { text: 'PISTOL!!', t: performance.now() }
    else if (maxStretch > 0.15) shout = { text: 'boing', t: performance.now() }
    maxStretch = 0
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

function updateHands(result) {
  pinchPoints = []
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

// ---------- physics ----------

function step() {
  const active = [...grabs.values()].map((g) => {
    const [ax, ay] = anchor(g)
    const dx = g.x - ax
    const dy = g.y - ay
    maxStretch = Math.max(maxStretch, Math.hypot(dx * aspect, dy))
    return { ax, ay, dx, dy, s2: 2 * g.sigma * g.sigma }
  })

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
    vel[i] = (vel[i] + (tx - offset[i]) * STIFFNESS) * DAMPING
    vel[i + 1] = (vel[i + 1] + (ty - offset[i + 1]) * STIFFNESS) * DAMPING
    offset[i] += vel[i]
    offset[i + 1] += vel[i + 1]
  }
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

function loop() {
  const now = performance.now()

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

  step()

  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video)
  gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuf)
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, offset)
  gl.clear(gl.DEPTH_BUFFER_BIT)
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0)

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
