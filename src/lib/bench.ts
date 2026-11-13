import * as ort from 'onnxruntime-web'
import type { BenchSettings, Fixture, FixtureSet, Manifest, ModelMeta, ModelResult } from './types'

/** Noise level used for the automatic robustness check. */
export const ROBUST_SIGMA = 0.05

const BASE = import.meta.env.BASE_URL

// Vite emits the onnxruntime Wasm binary as a hashed asset next to the bundle.
// Single-threaded so the numbers are comparable to a constrained edge core and
// work on static hosts without COOP/COEP headers.
ort.env.wasm.numThreads = 1
ort.env.logLevel = 'error'

export async function loadJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export const loadManifest = () => loadJSON<Manifest>('models/manifest.json')
export const loadFixtures = () => loadJSON<FixtureSet>('fixtures/road_vibration.json')

/** Edge Viability Score: (Accuracy x 100) / (Latency ms x Size MB). */
export function evs(accuracy: number, latencyMs: number, sizeMB: number): number {
  const lat = Math.max(latencyMs, 0.001)
  const size = Math.max(sizeMB, 0.001)
  return (accuracy * 100) / (lat * size)
}

/** Deterministic Gaussian noise so a given noise level is reproducible across models. */
function seededNormal(seed: number) {
  let s = seed >>> 0 || 1
  const u = () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) + 0.5) / 4294967296
  }
  return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u())
}

export function prepareInputs(samples: Fixture[], noise: number, seed = 1337): Float32Array[] {
  const rand = seededNormal(seed)
  return samples.map((s) => {
    const arr = new Float32Array(s.data.length)
    for (let i = 0; i < arr.length; i++) arr[i] = s.data[i] + (noise > 0 ? noise * rand() : 0)
    return arr
  })
}

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))
  return sorted[idx]
}

export function summarize(latencies: number[]) {
  const sorted = [...latencies].sort((a, b) => a - b)
  const mean = latencies.reduce((a, b) => a + b, 0) / Math.max(1, latencies.length)
  const variance = latencies.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, latencies.length)
  return {
    meanMs: mean,
    p50Ms: percentile(sorted, 50),
    p95Ms: percentile(sorted, 95),
    maxMs: sorted[sorted.length - 1] ?? 0,
    stdMs: Math.sqrt(variance),
  }
}

const urlFor = (m: ModelMeta) => (/^(blob:|https?:)/.test(m.file) ? m.file : `${BASE}${m.file}`)

/** Initialise the Wasm runtime once so the first model's cold start isn't charged for it. */
let runtimeReady: Promise<unknown> | null = null
export function warmRuntime(model: ModelMeta) {
  runtimeReady ??= ort.InferenceSession.create(urlFor(model), { executionProviders: ['wasm'] }).catch(() => null)
  return runtimeReady
}

interface Entry {
  session: ort.InferenceSession
  loadMs: number
  inName: string
  outName: string
  /** input tensor shape that the model accepted */
  shape: number[]
  /** output is logits, not probabilities */
  needsSoftmax: boolean
}

const sessions = new Map<string, Entry>()
const N_CLASSES = 4
const INPUT_SHAPES = [[1, 384], [1, 3, 128]]

export function softmax(v: number[]) {
  const m = Math.max(...v)
  const e = v.map((x) => Math.exp(x - m))
  const s = e.reduce((a, c) => a + c, 0)
  return e.map((x) => x / s)
}

/**
 * Create a session and work out how to talk to the model: which input shape it takes
 * ([1,384] flat or [1,3,128]) and which output holds one float per class.
 */
async function getSession(model: ModelMeta, backend: string): Promise<Entry> {
  const key = `${model.id}:${backend}`
  const cached = sessions.get(key)
  if (cached) return cached
  const t0 = performance.now()
  const session = await ort.InferenceSession.create(urlFor(model), {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  })
  const loadMs = performance.now() - t0
  const inName = session.inputNames[0]
  let lastErr: unknown
  for (const shape of INPUT_SHAPES) {
    try {
      const out = await session.run({ [inName]: new ort.Tensor('float32', new Float32Array(384).fill(0.5), shape) })
      const names = session.outputNames.filter((n) => {
        const t = out[n]
        return t && t.type === 'float32' && t.data.length === N_CLASSES
      })
      const outName = names.find((n) => n.toLowerCase().includes('prob')) ?? names[0]
      if (!outName) throw new Error(`no float output with ${N_CLASSES} values (one per class) was found`)
      const vals = Array.from(out[outName].data as Float32Array)
      const sum = vals.reduce((a, c) => a + c, 0)
      const needsSoftmax = Math.abs(sum - 1) > 0.01 || vals.some((v) => v < 0)
      const entry: Entry = { session, loadMs, inName, outName, shape, needsSoftmax }
      sessions.set(key, entry)
      return entry
    } catch (e) {
      lastErr = e
    }
  }
  throw new Error(`Model does not accept a [1,384] or [1,3,128] float32 input: ${lastErr instanceof Error ? lastErr.message : lastErr}`)
}

async function inferWith(e: Entry, x: Float32Array) {
  const out = await e.session.run({ [e.inName]: new ort.Tensor('float32', x, e.shape) }, [e.outName])
  const v = Array.from(out[e.outName].data as Float32Array)
  return e.needsSoftmax ? softmax(v) : v
}

/** Validate a user-supplied model. Throws a readable error if it cannot be benchmarked. */
export async function probeModel(model: ModelMeta) {
  const e = await getSession(model, 'wasm')
  return { shape: e.shape, output: e.outName, needsSoftmax: e.needsSoftmax }
}

/** One prediction on the Wasm backend; used by the signal lab. */
export async function predict(model: ModelMeta, x: Float32Array) {
  return inferWith(await getSession(model, 'wasm'), x)
}

function argmax(a: ArrayLike<number>, n: number) {
  let best = 0
  for (let i = 1; i < n; i++) if (a[i] > a[best]) best = i
  return best
}

export function macroF1(cm: number[][]) {
  const n = cm.length
  let total = 0
  for (let c = 0; c < n; c++) {
    const tp = cm[c][c]
    const fn = cm[c].reduce((a, v) => a + v, 0) - tp
    const fp = cm.reduce((a, row) => a + row[c], 0) - tp
    total += tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn)
  }
  return total / n
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

export type Phase = 'loading' | 'watching' | 'robustness' | 'timing' | 'done'

export interface VisualStep {
  modelId: string
  sampleIndex: number
  probs: number[]
  ms: number
}

export interface BenchCallbacks {
  onPhase?: (modelId: string, phase: Phase) => void
  onVisual?: (step: VisualStep) => void
  onProgress?: (modelId: string, pct: number) => void
}

export async function benchmarkModel(
  model: ModelMeta,
  samples: Fixture[],
  inputs: Float32Array[],
  nClasses: number,
  settings: BenchSettings,
  cb: BenchCallbacks = {},
): Promise<ModelResult> {
  cb.onPhase?.(model.id, 'loading')
  let backendUsed: string = settings.backend
  let sess: Entry
  try {
    sess = await getSession(model, backendUsed)
  } catch (e) {
    if (backendUsed === 'wasm') throw e
    // Tree ensembles, kNN and integer ops are Wasm-only kernels; fall back honestly.
    backendUsed = 'wasm'
    sess = await getSession(model, backendUsed)
  }
  const { session, loadMs } = sess
  const probName = sess.outName
  const feed = (x: Float32Array) => ({ [sess.inName]: new ort.Tensor('float32', x, sess.shape) })
  const infer = (x: Float32Array) => inferWith(sess, x)

  // 1. Visual pass: one real inference per sample, slowed down so people can watch it.
  cb.onPhase?.(model.id, 'watching')
  const probs: number[][] = []
  for (let i = 0; i < samples.length; i++) {
    const t0 = performance.now()
    const p = await infer(inputs[i])
    const ms = performance.now() - t0
    probs.push(p)
    if (settings.watchDelay > 0) {
      cb.onVisual?.({ modelId: model.id, sampleIndex: i, probs: p, ms })
      await sleep(settings.watchDelay)
    }
  }

  // 2. Robustness: the clean fixtures with a fixed amount of sensor noise (same for every model).
  cb.onPhase?.(model.id, 'robustness')
  const noisy = prepareInputs(samples, ROBUST_SIGMA, 4242)
  let noisyCorrect = 0
  for (let i = 0; i < samples.length; i++) {
    if (argmax(await infer(noisy[i]), nClasses) === samples[i].labelIndex) noisyCorrect++
  }

  // 3. Timing.
  cb.onPhase?.(model.id, 'timing')
  for (let w = 0; w < settings.warmup; w++) await session.run(feed(inputs[w % inputs.length]))

  // Browsers coarsen performance.now() (often to 0.1 ms without cross-origin isolation),
  // and these models run in microseconds. Calibrate a batch size so each timed sample
  // spans >= 2 ms, then report per-call latency = batch time / batch size.
  let inner = 1
  const probe = feed(inputs[0])
  while (inner < 512) {
    const t0 = performance.now()
    for (let k = 0; k < inner; k++) await session.run(probe, [probName])
    if (performance.now() - t0 >= 2) break
    inner *= 2
  }

  const latencies: number[] = []
  const total = settings.runs * samples.length
  for (let r = 0; r < settings.runs; r++) {
    for (let i = 0; i < samples.length; i++) {
      const f = feed(inputs[i])
      const t0 = performance.now()
      for (let k = 0; k < inner; k++) await session.run(f, [probName])
      latencies.push((performance.now() - t0) / inner)
    }
    cb.onProgress?.(model.id, Math.round((((r + 1) * samples.length) / total) * 100))
    await sleep(0) // keep the UI responsive between passes
  }

  const predictions = probs.map((p) => argmax(p, nClasses))
  const confidences = probs.map((p, i) => p[predictions[i]])
  const correct = predictions.reduce((a, p, i) => a + (p === samples[i].labelIndex ? 1 : 0), 0)
  const hazardIdx = samples.map((s, i) => (s.labelIndex >= 2 ? i : -1)).filter((i) => i >= 0)
  const hazardHit = hazardIdx.filter((i) => predictions[i] === samples[i].labelIndex).length
  const confusion = Array.from({ length: nClasses }, () => new Array<number>(nClasses).fill(0))
  predictions.forEach((p, i) => confusion[samples[i].labelIndex][p]++)
  cb.onPhase?.(model.id, 'done')
  return {
    modelId: model.id,
    backendUsed,
    latencies,
    predictions,
    confidences,
    probs: probs.map((p) => p.map((v) => +v.toFixed(4))),
    correct,
    total: samples.length,
    accuracy: correct / samples.length,
    macroF1: macroF1(confusion),
    confusion,
    hazardRecall: hazardIdx.length ? hazardHit / hazardIdx.length : 0,
    noisyAccuracy: noisyCorrect / samples.length,
    loadMs,
    callsPerTiming: inner,
    ...summarize(latencies),
  }
}
