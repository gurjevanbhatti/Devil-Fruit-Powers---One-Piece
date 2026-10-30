// devil fruits raining down behind the start screen

const canvas = document.getElementById('fruits')
const ctx = canvas.getContext('2d')

// the hand-drawn pixel-art devil fruit
const fruitImg = new Image()
fruitImg.src = 'images/fruit.png'

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches

let width = 0
let height = 0
let fruits = []

// the town, sea and ship are drawn small and blown up with sharp pixels, so they look
// like pixel art to match the fruit and hat
const PIXEL = 4 // how many screen pixels each background pixel takes up
const bg = document.createElement('canvas')
const bgCtx = bg.getContext('2d')

function spawn(fruit, anywhere) {
  // depth 0 is far away (small, slow, faded), 1 is close
  const depth = Math.random()
  fruit.size = 20 + depth * 30
  fruit.x = Math.random() * width
  fruit.y = anywhere ? Math.random() * height : -fruit.size
  fruit.vy = (40 + depth * 90) * (reduceMotion ? 0.3 : 1)
  fruit.rot = Math.random() * Math.PI * 2
  fruit.vr = (Math.random() - 0.5) * 1.6
  fruit.phase = Math.random() * Math.PI * 2
  fruit.alpha = 0.55 + depth * 0.45
  fruit.depth = depth
  return fruit
}

function resize() {
  const dpr = window.devicePixelRatio || 1
  width = canvas.clientWidth
  height = canvas.clientHeight
  canvas.width = width * dpr
  canvas.height = height * dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  // keep the pixel art sharp instead of blurry when it's scaled
  ctx.imageSmoothingEnabled = false
  bg.width = Math.ceil(width / PIXEL)
  bg.height = Math.ceil(height / PIXEL)
  buildTown(bg.width, bg.height)

  const count = Math.round(Math.min(18, Math.max(8, (width * height) / 60000)))
  while (fruits.length < count) fruits.push(spawn({}, true))
  fruits.length = count
  // far fruits first so the close ones are drawn on top
  fruits.sort((a, b) => a.depth - b.depth)
}

// ---------- the harbour town ----------

// a little town on a cliff above the sea (windmills, red roofs, trees, paths, a dock with
// the ship tied up), seen from above like a map. the town is laid out at random once,
// then only the sea, the windmills and the ship move

function random(seed) {
  // the same seed always gives the same town, so it doesn't change on every resize
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const ROOFS = [
  ['#d9472b', '#a8321c'],
  ['#e8743a', '#b8541f'],
  ['#e9b23c', '#b8861f'],
]

// the town is drawn once into here, then copied each frame
const town = document.createElement('canvas')
const townCtx = town.getContext('2d')
let mills = [] // windmill hubs, their blades are drawn every frame
let sparkles = [] // glints on the water
let dock = null

// where the land ends, going down a little to the left
function coastY(x, W, H) {
  return H * (0.52 - 0.1 * (x / W)) + Math.sin(x * 0.09) * 1.5 + Math.sin(x * 0.023 + 1) * 4
}

function house(g, x, y, u, rand) {
  const [roof, roofDark] = ROOFS[Math.floor(rand() * ROOFS.length)]
  const w = (9 + Math.floor(rand() * 4)) * u
  const h = 5 * u
  const r = 4 * u
  g.fillStyle = 'rgba(0, 0, 0, 0.2)'
  g.fillRect(x + u, y + r + h, w, u) // shadow
  g.fillStyle = '#f3ead6'
  g.fillRect(x, y + r, w, h) // walls
  g.fillStyle = '#d6c9ad'
  g.fillRect(x + w - 2 * u, y + r, 2 * u, h)
  g.fillStyle = roof
  g.fillRect(x - u, y, w + 2 * u, r)
  g.fillStyle = roofDark
  g.fillRect(x - u, y + r - u, w + 2 * u, u)
  g.fillStyle = 'rgba(255, 255, 255, 0.35)'
  g.fillRect(x - u, y, w + 2 * u, u)
  g.fillStyle = '#7a4a2a'
  g.fillRect(x + 2 * u, y + r + h - 3 * u, 2 * u, 3 * u) // door
  g.fillStyle = '#5b7fa8'
  g.fillRect(x + 5 * u, y + r + u, 2 * u, 2 * u) // window
}

function tree(g, x, y, u, rand) {
  const r = (2.5 + rand() * 1.5) * u
  g.fillStyle = 'rgba(0, 0, 0, 0.2)'
  g.beginPath()
  g.ellipse(x + u, y + r * 0.9, r, r * 0.45, 0, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = '#2c6e31'
  g.beginPath()
  g.arc(x, y, r, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = '#4a9a3c'
  g.beginPath()
  g.arc(x - r * 0.3, y - r * 0.3, r * 0.5, 0, Math.PI * 2)
  g.fill()
}

function millTower(g, x, y, u) {
  g.fillStyle = 'rgba(0, 0, 0, 0.2)'
  g.fillRect(x + u, y + 12 * u, 6 * u, u)
  g.fillStyle = '#efeee8'
  g.fillRect(x, y + 3 * u, 6 * u, 9 * u)
  g.fillStyle = '#cfcfc6'
  g.fillRect(x + 4 * u, y + 3 * u, 2 * u, 9 * u)
  g.fillStyle = '#3f8f4a'
  g.fillRect(x - u, y + u, 8 * u, 3 * u) // green cap
  g.fillRect(x, y, 6 * u, u)
  g.fillStyle = '#7a4a2a'
  g.fillRect(x + 2 * u, y + 9 * u, 2 * u, 3 * u)
}

function buildTown(W, H) {
  town.width = W
  town.height = H
  const g = townCtx
  const rand = random(7)
  const u = Math.max(1, Math.round(Math.min(W, H) / 170))

  // sea, darker further out
  const sea = ['#2f86e0', '#2879d4', '#226cc6', '#1d60b8', '#1855a8', '#144a98']
  for (let y = 0; y < H; y++) {
    g.fillStyle = sea[Math.min(sea.length - 1, Math.floor(((y / H) * sea.length * 1.3) | 0))]
    g.fillRect(0, y, W, 1)
  }

  // land: rocky cliff on the left, sandy beach on the right where the dock is
  for (let x = 0; x < W; x++) {
    const c = Math.round(coastY(x, W, H))
    const beach = x > W * 0.55
    g.fillStyle = '#5aa83c'
    g.fillRect(x, 0, 1, c)
    g.fillStyle = beach ? '#e8d39a' : '#8d8f8a'
    g.fillRect(x, c, 1, (beach ? 3 : 4) * u)
    g.fillStyle = beach ? '#cdb77c' : '#6b6d69'
    g.fillRect(x, c + (beach ? 3 : 4) * u, 1, u)
  }
  // patchy grass
  for (let i = 0; i < (W * H) / 60; i++) {
    const x = Math.floor(rand() * W)
    const y = Math.floor(rand() * coastY(x, W, H)) - 2
    g.fillStyle = rand() < 0.5 ? '#6fbf4a' : '#4a9432'
    g.fillRect(x, y, u, u)
  }

  // dirt paths winding down to the dock
  // the ship sits against the right edge with the dock just to its left
  const shipX = W - shipSize(W) * 1.25
  const dockX = Math.round(shipX - shipSize(W) * 1.2 - 4 * u)
  dock = { x: dockX, y: coastY(dockX, W, H), shipX }
  g.strokeStyle = '#d9c08a'
  g.lineWidth = 3 * u
  g.lineCap = 'round'
  g.beginPath()
  g.moveTo(0, H * 0.18)
  g.bezierCurveTo(W * 0.35, H * 0.12, W * 0.45, H * 0.3, dock.x, dock.y)
  g.stroke()

  // houses, windmills and trees, spaced out, drawn top to bottom so nearer ones overlap
  const things = []
  const free = (x, y, gap) => things.every((t) => Math.hypot(t.x - x, t.y - y) > gap + t.gap)
  // sides only keeps something out from behind the card in the middle
  const place = (kind, count, gap, edge, sidesOnly) => {
    for (let n = 0, tries = 0; n < count && tries < count * 40; tries++) {
      let x = rand() * (W - 12 * u)
      if (sidesOnly) x = rand() < 0.5 ? rand() * W * 0.22 : W * 0.78 + rand() * (W * 0.22 - 10 * u)
      const y = rand() * (coastY(x, W, H) - edge * u)
      if (y < -4 * u || !free(x, y, gap * u)) continue
      things.push({ kind, x, y, gap: gap * u })
      n++
    }
  }
  place('mill', 2, 10, 16, true)
  place('house', Math.round((W * H) / 6000), 14, 12)
  place('tree', Math.round((W * H) / 1400), 5, 6)
  things.sort((a, b) => a.y - b.y)
  mills = []
  for (const t of things) {
    const x = Math.round(t.x)
    const y = Math.round(t.y)
    if (t.kind === 'house') house(g, x, y, u, rand)
    else if (t.kind === 'tree') tree(g, x, y, u, rand)
    else {
      millTower(g, x, y, u)
      mills.push({ x: x + 3 * u, y: y + 2 * u, len: 7 * u, spin: rand() * 6, u })
    }
  }

  // the dock, with a rowboat tied to it
  g.fillStyle = '#8b5a2b'
  g.fillRect(Math.round(dock.x), Math.round(dock.y), 4 * u, 16 * u)
  g.fillStyle = '#6b4226'
  for (let i = 0; i < 16; i += 2) g.fillRect(Math.round(dock.x), Math.round(dock.y) + i * u, 4 * u, u / 2 || 1)
  g.fillStyle = '#a0683a'
  g.fillRect(Math.round(dock.x) - 7 * u, Math.round(dock.y) + 9 * u, 5 * u, 2 * u)

  // glints on the water, they twinkle every frame
  sparkles = []
  for (let i = 0; i < (W * H) / 500; i++) {
    const x = rand() * W
    const y = coastY(x, W, H) + 6 * u + rand() * H
    if (y < H) sparkles.push({ x, y, len: (2 + Math.floor(rand() * 4)) * u, phase: rand() * 6 })
  }
}

function drawTown(g, W, H, t) {
  g.drawImage(town, 0, 0)

  // twinkling glints, drifting slowly
  for (const s of sparkles) {
    const on = Math.sin(t * 1.5 + s.phase)
    if (on < 0.2) continue
    g.fillStyle = on > 0.8 ? '#cfeaff' : '#7fc0f5'
    g.fillRect(Math.round((s.x + t * 2) % W), Math.round(s.y), s.len, 1)
  }

  // foam washing along the shore
  g.fillStyle = 'rgba(255, 255, 255, 0.8)'
  const wash = Math.floor(t * 4)
  for (let x = 0; x < W; x++) {
    if ((x + wash) % 9 < 5) g.fillRect(x, Math.round(coastY(x, W, H)) + 5, 1, 1)
  }

  // windmill blades going round
  g.fillStyle = '#f7f5ec'
  for (const m of mills) {
    const a0 = (reduceMotion ? 0 : t * 1.2) + m.spin
    for (let k = 0; k < 4; k++) {
      const a = a0 + (k * Math.PI) / 2
      for (let d = m.u; d <= m.len; d++) {
        g.fillRect(Math.round(m.x + Math.cos(a) * d), Math.round(m.y + Math.sin(a) * d), m.u, m.u)
      }
    }
    g.fillStyle = '#7a4a2a'
    g.fillRect(m.x, m.y, m.u, m.u)
    g.fillStyle = '#f7f5ec'
  }

  // the ship, tied up next to the dock and rocking gently
  const s = shipSize(W)
  const bob = reduceMotion ? 0 : Math.sin(t * 1.6)
  drawShip(g, dock.shipX, dock.y + s * 1.3, s, bob)
}

function shipSize(W) {
  return Math.min(70, Math.max(40, W * PIXEL * 0.06)) / PIXEL
}

function drawShip(ctx, x, y, s, bob) {
  // a little pirate ship with a sheep figurehead and a straw hat flag. s is its size,
  // everything is drawn in units of s
  ctx.save()
  ctx.translate(Math.round(x), Math.round(y + bob * 0.6))
  ctx.rotate(bob * 0.04)
  ctx.scale(s, s)
  ctx.lineJoin = 'round'
  ctx.lineWidth = 0.05
  ctx.strokeStyle = '#3a2214'

  // mast
  ctx.fillStyle = '#6b4226'
  ctx.fillRect(-0.04, -1.55, 0.08, 1.4)
  ctx.strokeRect(-0.04, -1.55, 0.08, 1.4)

  // sail, puffed out by the wind
  ctx.beginPath()
  ctx.moveTo(-0.55, -1.3)
  ctx.quadraticCurveTo(0, -1.45, 0.55, -1.3)
  ctx.quadraticCurveTo(0.68, -0.85, 0.55, -0.42)
  ctx.quadraticCurveTo(0, -0.55, -0.55, -0.42)
  ctx.quadraticCurveTo(-0.42, -0.85, -0.55, -1.3)
  ctx.fillStyle = '#fdf6e3'
  ctx.fill()
  ctx.stroke()

  // flag: a skull wearing a straw hat
  ctx.fillStyle = '#1b1b1b'
  ctx.beginPath()
  ctx.moveTo(0.04, -1.55)
  ctx.lineTo(0.5, -1.55)
  ctx.lineTo(0.44, -1.47)
  ctx.lineTo(0.5, -1.27)
  ctx.lineTo(0.04, -1.27)
  ctx.closePath()
  ctx.fill()
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.arc(0.25, -1.38, 0.07, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = '#f4c430'
  ctx.fillRect(0.14, -1.47, 0.22, 0.04)
  ctx.fillStyle = '#e63946'
  ctx.fillRect(0.19, -1.45, 0.12, 0.02)

  // hull
  ctx.beginPath()
  ctx.moveTo(-1, -0.28)
  ctx.lineTo(0.95, -0.22)
  ctx.quadraticCurveTo(0.9, 0.12, 0.62, 0.22)
  ctx.lineTo(-0.72, 0.22)
  ctx.quadraticCurveTo(-0.98, 0.05, -1, -0.28)
  ctx.fillStyle = '#c98b4a'
  ctx.fill()
  ctx.stroke()
  // a darker stripe and a couple of portholes
  ctx.fillStyle = '#8b5a2b'
  ctx.fillRect(-0.9, -0.08, 1.72, 0.08)
  ctx.fillStyle = '#3a2214'
  for (const x of [-0.45, 0, 0.45]) {
    ctx.beginPath()
    ctx.arc(x, 0.09, 0.045, 0, Math.PI * 2)
    ctx.fill()
  }

  // sheep figurehead on the front
  ctx.fillStyle = '#fdf6e3'
  ctx.beginPath()
  ctx.arc(1.02, -0.36, 0.15, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(1.0, -0.36, 0.07, 0.3, Math.PI * 1.8)
  ctx.stroke()
  ctx.fillStyle = '#3a2214'
  ctx.beginPath()
  ctx.arc(1.1, -0.38, 0.025, 0, Math.PI * 2)
  ctx.fill()

  ctx.restore()

  // the sea in front of the bottom of the hull, so it sits in the water
  ctx.fillStyle = '#2879d4'
  ctx.fillRect(Math.round(x - s * 1.2), Math.round(y + s * 0.22 + bob * 0.6), Math.round(s * 2.4), Math.ceil(s * 0.15))
}

let last = performance.now()

function frame(now) {
  // the start screen gets removed once the camera is on, so stop drawing
  if (!canvas.isConnected) return
  const dt = Math.min(0.05, (now - last) / 1000)
  last = now

  // background small, then blown up with sharp pixels
  drawTown(bgCtx, bg.width, bg.height, now / 1000)
  ctx.drawImage(bg, 0, 0, bg.width * PIXEL, bg.height * PIXEL)
  for (const f of fruits) {
    f.y += f.vy * dt
    f.rot += f.vr * dt
    f.phase += dt
    if (f.y > height + f.size) spawn(f, false)

    const sway = Math.sin(f.phase * 1.3) * 18 * (0.5 + f.depth)
    ctx.save()
    ctx.globalAlpha = f.alpha
    ctx.translate(f.x + sway, f.y)
    ctx.rotate(f.rot)
    // the drawing is a little taller than it is wide
    const h = f.size * (fruitImg.naturalHeight / fruitImg.naturalWidth || 1)
    if (fruitImg.complete) ctx.drawImage(fruitImg, -f.size / 2, -h / 2, f.size, h)
    ctx.restore()
  }
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
resize()
requestAnimationFrame(frame)
