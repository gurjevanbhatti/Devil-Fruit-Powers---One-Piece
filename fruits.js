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

  const count = Math.round(Math.min(18, Math.max(8, (width * height) / 60000)))
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
