// swirly fruits raining down behind the start screen

const canvas = document.getElementById('fruits')
const ctx = canvas.getContext('2d')

const COLORS = ['#8e44ad', '#ff8c1a', '#4caf50', '#ff5fa2', '#3a7bd5', '#e63946', '#f4c430']
const KINDS = ['round', 'pear', 'long']
const SPRITE = 128

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16)
  const c = (v) => Math.max(0, Math.min(255, Math.round(v + amount * 255)))
  return `rgb(${c(n >> 16)}, ${c((n >> 8) & 255)}, ${c(n & 255)})`
}

function bodyPath(g, kind) {
  g.beginPath()
  if (kind === 'round') {
    g.arc(64, 72, 44, 0, Math.PI * 2)
  } else if (kind === 'pear') {
    g.moveTo(64, 26)
    g.bezierCurveTo(88, 26, 84, 56, 100, 76)
    g.bezierCurveTo(116, 100, 96, 120, 64, 120)
    g.bezierCurveTo(32, 120, 12, 100, 28, 76)
    g.bezierCurveTo(44, 56, 40, 26, 64, 26)
  } else {
    g.ellipse(64, 74, 30, 48, 0.2, 0, Math.PI * 2)
  }
}

function spiral(g, cx, cy, turns, spacing) {
  g.beginPath()
  for (let t = 0; t <= turns * Math.PI * 2; t += 0.2) {
    const r = spacing * t
    const x = cx + Math.cos(t) * r
    const y = cy + Math.sin(t) * r
    t === 0 ? g.moveTo(x, y) : g.lineTo(x, y)
  }
  g.stroke()
}

// draw each fruit once to its own canvas, then just stamp it every frame
function makeSprite(kind, color) {
  const c = document.createElement('canvas')
  c.width = c.height = SPRITE
  const g = c.getContext('2d')

  const fill = g.createRadialGradient(50, 55, 6, 64, 75, 60)
  fill.addColorStop(0, shade(color, 0.25))
  fill.addColorStop(1, shade(color, -0.2))
  bodyPath(g, kind)
  g.fillStyle = fill
  g.fill()

  // the swirl pattern
  g.save()
  bodyPath(g, kind)
  g.clip()
  g.strokeStyle = shade(color, -0.35)
  g.lineWidth = 3.5
  g.lineCap = 'round'
  for (const [x, y] of [[44, 58], [82, 66], [56, 98], [90, 104], [30, 92]]) {
    spiral(g, x, y, 2.2, 1.6)
  }
  g.restore()

  bodyPath(g, kind)
  g.strokeStyle = '#3a2214'
  g.lineWidth = 3
  g.stroke()

  // shine
  g.fillStyle = 'rgba(255, 255, 255, 0.35)'
  g.beginPath()
  g.ellipse(46, 54, 9, 14, -0.5, 0, Math.PI * 2)
  g.fill()

  // stem and leaf
  g.strokeStyle = '#5b3a1e'
  g.lineWidth = 5
  g.beginPath()
  g.moveTo(64, 30)
  g.quadraticCurveTo(62, 16, 70, 8)
  g.stroke()
  g.fillStyle = '#3f9a3a'
  g.strokeStyle = '#1f5a1c'
  g.lineWidth = 2
  g.beginPath()
  g.ellipse(82, 16, 14, 6, -0.4, 0, Math.PI * 2)
  g.fill()
  g.stroke()

  return c
}

const sprites = []
for (const kind of KINDS) for (const color of COLORS) sprites.push(makeSprite(kind, color))

let width = 0
let height = 0
let fruits = []

function spawn(fruit, anywhere) {
  // depth 0 is far away (small, slow, faded), 1 is close
  const depth = Math.random()
  fruit.sprite = sprites[Math.floor(Math.random() * sprites.length)]
  fruit.size = 36 + depth * 60
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

  const count = Math.round(Math.min(40, Math.max(12, (width * height) / 26000)))
  while (fruits.length < count) fruits.push(spawn({}, true))
  fruits.length = count
  // far fruits first so the close ones are drawn on top
  fruits.sort((a, b) => a.depth - b.depth)
}

let last = performance.now()

function frame(now) {
  // the start screen gets removed once the camera is on, so stop drawing
  if (!canvas.isConnected) return
  const dt = Math.min(0.05, (now - last) / 1000)
  last = now

  ctx.clearRect(0, 0, width, height)
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
    ctx.drawImage(f.sprite, -f.size / 2, -f.size / 2, f.size, f.size)
    ctx.restore()
  }
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
resize()
requestAnimationFrame(frame)
