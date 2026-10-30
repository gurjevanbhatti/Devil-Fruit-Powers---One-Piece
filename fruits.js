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

  const count = Math.round(Math.min(40, Math.max(12, (width * height) / 26000)))
  while (fruits.length < count) fruits.push(spawn({}, true))
  fruits.length = count
  // far fruits first so the close ones are drawn on top
  fruits.sort((a, b) => a.depth - b.depth)
}

// ---------- the sea and the ship ----------

// where the sky meets the sea, matching the background gradient in style.css
const HORIZON = 0.585
const SHIP_SPEED = 22 // pixels per second

let shipX = 120

function drawWaves(t) {
  // a few soft wave lines drifting across the sea
  ctx.save()
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)'
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  const top = height * HORIZON
  for (let row = 0; row < 6; row++) {
    const y = top + 14 + row * ((height - top) / 6)
    const gap = 140 + row * 30
    const drift = (t * (8 + row * 4)) % gap
    for (let x = -gap + drift + (row % 2) * 60; x < width + gap; x += gap) {
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.quadraticCurveTo(x + 12, y - 6, x + 24, y)
      ctx.quadraticCurveTo(x + 36, y + 6, x + 48, y)
      ctx.stroke()
    }
  }
  ctx.restore()
}

function drawShip(t, dt) {
  // a little pirate ship with a sheep figurehead and a straw hat flag, sailing along
  // the horizon and bobbing on the waves. s is its size, everything is drawn in units of s
  const s = Math.min(70, Math.max(40, width * 0.05))
  if (!reduceMotion) shipX += SHIP_SPEED * dt
  if (shipX > width + s * 2) shipX = -s * 2
  const bob = reduceMotion ? 0 : Math.sin(t * 1.6)

  ctx.save()
  ctx.translate(shipX, height * HORIZON + s * 0.08 + bob * 2)
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
  ctx.fillStyle = '#2f9be0'
  ctx.fillRect(shipX - s * 1.2, height * HORIZON + s * 0.22 + bob * 2, s * 2.4, s * 0.15)
}

let last = performance.now()

function frame(now) {
  // the start screen gets removed once the camera is on, so stop drawing
  if (!canvas.isConnected) return
  const dt = Math.min(0.05, (now - last) / 1000)
  last = now

  ctx.clearRect(0, 0, width, height)
  drawWaves(now / 1000)
  drawShip(now / 1000, dt)
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
