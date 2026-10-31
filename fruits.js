// devil fruits raining down behind the start screen

const canvas = document.getElementById('fruits')
const ctx = canvas.getContext('2d')

// the hand-drawn pixel-art devil fruit, in a different colour for the rain so it
// doesn't look like the real one in the chest
const RAIN_HUE = 80 // degrees round the colour wheel from the real purple
const fruitImg = new Image()
let rainFruit = null
fruitImg.onload = () => {
  rainFruit = document.createElement('canvas')
  rainFruit.width = fruitImg.naturalWidth
  rainFruit.height = fruitImg.naturalHeight
  const c = rainFruit.getContext('2d')
  c.drawImage(fruitImg, 0, 0)
  const img = c.getImageData(0, 0, rainFruit.width, rainFruit.height)
  const d = img.data
  for (let i = 0; i < d.length; i += 4) {
    // only the purple fruit, so the stem stays the same
    if (d[i + 2] <= d[i] || d[i + 2] <= d[i + 1]) continue
    const [r, g, b] = hueShift(d[i], d[i + 1], d[i + 2], RAIN_HUE)
    d[i] = r
    d[i + 1] = g
    d[i + 2] = b
  }
  c.putImageData(img, 0, 0)
}
fruitImg.src = 'images/fruit.png'

function hueShift(r, g, b, deg) {
  // turn the colour round the colour wheel, keeping how light it is
  const a = (deg * Math.PI) / 180
  const cos = Math.cos(a)
  const sin = Math.sin(a)
  const k = 1 / 3
  const sq = Math.sqrt(k)
  const m0 = cos + (1 - cos) * k
  const m1 = k * (1 - cos) - sq * sin
  const m2 = k * (1 - cos) + sq * sin
  const clamp = (v) => Math.max(0, Math.min(255, v))
  return [
    clamp(r * m0 + g * m1 + b * m2),
    clamp(r * m2 + g * m0 + b * m1),
    clamp(r * m1 + g * m2 + b * m0),
  ]
}

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
    if (rainFruit) ctx.drawImage(rainFruit, -f.size / 2, -h / 2, f.size, h)
    ctx.restore()
  }
  requestAnimationFrame(frame)
}

window.addEventListener('resize', resize)
resize()
requestAnimationFrame(frame)
