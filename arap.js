// As-rigid-as-possible shape manipulation (Igarashi, Moscovich & Hughes, SIGGRAPH 2005).
// Bends a triangle mesh so every triangle stays as close to its rest shape as it can
// while a few handle vertices get dragged around. Used to stretch the real picture of a
// finger like rubber. Two steps, both set up once per grab so each frame is cheap:
//   1. let each triangle rotate and scale freely to fit the handles
//   2. take just the rotation from step 1 and fit the rest-length edges to it

// ---------- small dense linear algebra ----------

function transpose(M) {
  return M[0].map((_, j) => M.map((row) => row[j]))
}

function matMul(A, B) {
  return A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)))
}

function invert(M) {
  // gauss-jordan, fine for the 4x4s here
  const n = M.length
  const A = M.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r
    ;[A[c], A[p]] = [A[p], A[c]]
    const d = A[c][c] || 1e-12
    for (let j = 0; j < 2 * n; j++) A[c][j] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = A[r][c]
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]
    }
  }
  return A.map((row) => row.slice(n))
}

// A is n*n, symmetric positive definite. returns the lower triangle L with A = L Lᵀ
function cholesky(A, n) {
  const L = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j]
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j]
    }
  }
  return L
}

function cholSolve(L, n, b) {
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]
    y[i] = s / L[i * n + i]
  }
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]
    x[i] = s / L[i * n + i]
  }
  return x
}

// ---------- the solver ----------

export class ARAP {
  // verts: [[x, y]] rest positions, tris: [[i, j, k]], handles: vertex indices you'll drag
  constructor(verts, tris, handles, weight = 1000) {
    const n = verts.length
    const N = 2 * n
    this.n = n
    this.tris = tris
    this.w = weight

    // step 1: how far each triangle is from being a similarity transform of its rest shape
    const A1 = new Float64Array(N * N)
    this.ginv = []
    this.dofs = []
    for (const t of tris) {
      const G = []
      for (const v of t) {
        const [px, py] = verts[v]
        G.push([px, -py, 1, 0], [py, px, 0, 1])
      }
      const Gt = transpose(G)
      const ginv = matMul(invert(matMul(Gt, G)), Gt) // pulls (a, b, tx, ty) out of the 6 coords
      const H = matMul(G, ginv)
      const F = H.map((row, i) => row.map((h, j) => (i === j ? 1 : 0) - h))
      const FtF = matMul(transpose(F), F)
      const dofs = t.flatMap((v) => [2 * v, 2 * v + 1])
      for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) A1[dofs[a] * N + dofs[b]] += FtF[a][b]
      this.ginv.push(ginv)
      this.dofs.push(dofs)
    }
    for (const h of handles) {
      A1[2 * h * N + 2 * h] += weight
      A1[(2 * h + 1) * N + 2 * h + 1] += weight
    }
    for (let i = 0; i < N; i++) A1[i * N + i] += 1e-8
    this.L1 = cholesky(A1, N)

    // step 2: every triangle edge, which should keep its rest length (x and y solve separately)
    const A2 = new Float64Array(n * n)
    this.edges = []
    this.restEdges = []
    for (const t of tris) {
      for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
        this.edges.push([a, b])
        this.restEdges.push([verts[b][0] - verts[a][0], verts[b][1] - verts[a][1]])
        A2[a * n + a] += 1
        A2[b * n + b] += 1
        A2[a * n + b] -= 1
        A2[b * n + a] -= 1
      }
    }
    for (const h of handles) A2[h * n + h] += weight
    for (let i = 0; i < n; i++) A2[i * n + i] += 1e-8
    this.L2 = cholesky(A2, n)
  }

  // targets: Map of handle vertex -> [x, y]. returns the bent vertices
  solve(targets) {
    const { n, w } = this
    const N = 2 * n

    const b1 = new Float64Array(N)
    for (const [h, [x, y]] of targets) {
      b1[2 * h] = w * x
      b1[2 * h + 1] = w * y
    }
    const u = cholSolve(this.L1, N, b1)

    // each triangle's rotation from step 1 (scale thrown away), applied to its rest edges
    const bx = new Float64Array(n)
    const by = new Float64Array(n)
    let e = 0
    this.tris.forEach((_, ti) => {
      const V = this.dofs[ti].map((d) => u[d])
      const g = this.ginv[ti]
      let a = 0
      let b = 0
      for (let k = 0; k < 6; k++) {
        a += g[0][k] * V[k]
        b += g[1][k] * V[k]
      }
      const s = Math.hypot(a, b) || 1
      const c = a / s
      const sn = b / s
      for (let k = 0; k < 3; k++, e++) {
        const [ex, ey] = this.restEdges[e]
        const tx = c * ex - sn * ey
        const ty = sn * ex + c * ey
        const [i, j] = this.edges[e]
        bx[j] += tx
        bx[i] -= tx
        by[j] += ty
        by[i] -= ty
      }
    })
    for (const [h, [x, y]] of targets) {
      bx[h] += w * x
      by[h] += w * y
    }
    const x = cholSolve(this.L2, n, bx)
    const y = cholSolve(this.L2, n, by)
    return Array.from({ length: n }, (_, i) => [x[i], y[i]])
  }
}

// a springy 0 -> 1 that shoots a little past 1, wobbles once, and settles
export function easeSpring(t, overshoot = 1.9, wobble = 2.4) {
  if (t <= 0) return 0
  if (t >= 1) return 1
  const damp = 3 / Math.max(0.3, overshoot)
  return 1 - Math.exp(-damp * t) * Math.cos(wobble * Math.PI * t)
}
