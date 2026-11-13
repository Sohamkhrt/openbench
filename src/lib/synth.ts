/**
 * Road-vibration generator: a TypeScript port of the physical model in training/train.py,
 * with the random draws replaced by explicit knobs so people can shape a signal by hand.
 * Output layout matches the models' input: [x(128), y(128), z(128)] in g at 100 Hz.
 */
export const FS = 100
export const N = 128

export interface RoadParams {
  /** extra road texture, 0 (fresh asphalt) to 0.25 (gravel) */
  roughness: number
  /** pothole depth in g, 0 = none */
  pothole: number
  /** speed-bump height in g, 0 = none */
  bump: number
  /** vehicle speed in km/h */
  speed: number
  /** Gaussian sensor noise in g */
  noise: number
  seed: number
}

export const PRESETS: Record<string, Omit<RoadParams, 'seed'>> = {
  smooth: { roughness: 0, pothole: 0, bump: 0, speed: 40, noise: 0 },
  rough: { roughness: 0.15, pothole: 0, bump: 0, speed: 40, noise: 0 },
  pothole: { roughness: 0, pothole: 0.6, bump: 0, speed: 40, noise: 0 },
  speed_bump: { roughness: 0, pothole: 0, bump: 0.4, speed: 25, noise: 0 },
}

function rng(seed: number) {
  let s = (seed * 2654435761) >>> 0 || 1
  const u = () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) + 0.5) / 4294967296
  }
  const normal = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
  return { u, normal, uniform: (a: number, b: number) => a + (b - a) * u() }
}

type Rng = ReturnType<typeof rng>

/** 1/f^alpha noise with unit variance scaled to `scale` (inverse real DFT, like numpy irfft). */
function coloredNoise(r: Rng, alpha: number, scale: number) {
  const x = new Float64Array(N)
  if (scale <= 0) return x
  for (let k = 1; k <= N / 2; k++) {
    const f = (k * FS) / N
    const amp = 1 / Math.pow(f, alpha / 2)
    const re = r.normal() * amp
    const im = r.normal() * amp
    for (let n = 0; n < N; n++) {
      const ph = (2 * Math.PI * k * n) / N
      x[n] += re * Math.cos(ph) - im * Math.sin(ph)
    }
  }
  let mean = 0
  for (const v of x) mean += v
  mean /= N
  let sd = 0
  for (const v of x) sd += (v - mean) ** 2
  sd = Math.sqrt(sd / N) + 1e-9
  for (let n = 0; n < N; n++) x[n] = (scale * x[n]) / sd
  return x
}

const t = Float64Array.from({ length: N }, (_, i) => i / FS)
const gauss = (tt: number, t0: number, w: number) => Math.exp(-(((tt - t0) / w) ** 2))
/** Damped oscillation starting at t0 (suspension / wheel-hop response). */
function damped(tt: number, t0: number, f: number, zeta: number, amp: number) {
  if (tt < t0) return 0
  const w = 2 * Math.PI * f
  return amp * Math.exp(-zeta * w * (tt - t0)) * Math.sin(w * (tt - t0))
}

export function synthesize(p: RoadParams): Float32Array {
  const r = rng(p.seed)
  const base = 0.04
  const x = coloredNoise(r, 1.2, base * 0.6)
  const y = coloredNoise(r, 1.2, base * 0.6)
  const z = coloredNoise(r, 1.0, base)
  const engineF = r.uniform(22, 40)
  const engineA = r.uniform(0.01, 0.03)
  const phase = r.uniform(0, 6.28)
  const rz = coloredNoise(r, 0.6, p.roughness)
  const rx = coloredNoise(r, 0.6, p.roughness * 0.4)
  const ry = coloredNoise(r, 0.6, p.roughness * 0.4)
  const hop = r.uniform(9, 14)
  const body = r.uniform(1.2, 2.0)
  const lateral = r.uniform(6, 10)
  const axleDelay = 2.6 / (p.speed / 3.6) // wheelbase 2.6 m

  const out = new Float32Array(3 * N)
  for (let n = 0; n < N; n++) {
    const tt = t[n]
    const eng = engineA * Math.sin(2 * Math.PI * engineF * tt + phase)
    let vx = x[n] + eng * 0.5 + rx[n]
    const vy = y[n] + eng * 0.5 + ry[n]
    let vz = z[n] + eng + rz[n]

    if (p.pothole > 0) {
      const t0 = 0.35
      const d = p.pothole
      vz += -d * gauss(tt, t0, 0.025)
      vz += damped(tt, t0 + 0.04, hop, 0.25, d * 1.4)
      vz += damped(tt, t0 + 0.04, body, 0.3, d * 0.3)
      vx += damped(tt, t0 + 0.04, lateral, 0.3, d * 0.35)
      if (t0 + axleDelay < t[N - 1]) vz += damped(tt, t0 + axleDelay, hop, 0.25, d * 0.8)
    }
    if (p.bump > 0) {
      const t0 = 0.3
      const width = 0.2 * (30 / p.speed)
      const h = p.bump
      vz += h * gauss(tt, t0, width)
      vz += damped(tt, t0 + width, 1.4, 0.2, h * 0.5)
      if (t0 + axleDelay < t[N - 1]) vz += 0.8 * h * gauss(tt, t0 + axleDelay, width)
      // derivative of the bump profile couples into the forward axis
      vx += 0.15 * h * ((-2 * (tt - t0)) / (width * width)) * gauss(tt, t0, width) * 0.1
    }
    out[n] = vx + p.noise * r.normal()
    out[N + n] = vy + p.noise * r.normal()
    out[2 * N + n] = vz + 1 + p.noise * r.normal()
  }
  return out
}

export interface MotionSample {
  /** ms */
  t: number
  x: number
  y: number
  z: number
}

/**
 * Turn raw devicemotion samples (irregular, usually ~60 Hz, m/s² including gravity)
 * into the models' input: 128 samples at 100 Hz in g, with gravity on +z.
 */
export function resampleMotion(samples: MotionSample[]): Float32Array {
  if (samples.length < 2) throw new Error('not enough motion samples')
  const G = 9.80665
  const t0 = samples[0].t
  const span = samples[samples.length - 1].t - t0
  const meanZ = samples.reduce((a, s) => a + s.z, 0) / samples.length
  const sign = meanZ < 0 ? -1 : 1 // iOS and Android disagree on the sign of gravity
  const out = new Float32Array(3 * N)
  let j = 0
  for (let n = 0; n < N; n++) {
    const tt = t0 + Math.min(span, (n * 1000) / FS)
    while (j < samples.length - 2 && samples[j + 1].t < tt) j++
    const a = samples[j]
    const b = samples[j + 1]
    const w = b.t === a.t ? 0 : Math.min(1, Math.max(0, (tt - a.t) / (b.t - a.t)))
    out[n] = (sign * (a.x + (b.x - a.x) * w)) / G
    out[N + n] = (sign * (a.y + (b.y - a.y) * w)) / G
    out[2 * N + n] = (sign * (a.z + (b.z - a.z) * w)) / G
  }
  return out
}
