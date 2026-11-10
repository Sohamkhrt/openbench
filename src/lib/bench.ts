import * as ort from 'onnxruntime-web'
import type { BenchSettings, Fixture, FixtureSet, Manifest, ModelMeta, ModelResult } from './types'

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

export function prepareInputs(samples: Fixture[], noise: number): Float32Array[] {
  const rand = seededNormal(1337)
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

const sessions = new Map<string, ort.InferenceSession>()

async function getSession(model: ModelMeta, backend: string) {
  const key = `${model.id}:${backend}`
  const cached = sessions.get(key)
  if (cached) return { session: cached, loadMs: 0 }
  const t0 = performance.now()
  const session = await ort.InferenceSession.create(`${BASE}${model.file}`, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  })
  const loadMs = performance.now() - t0
  sessions.set(key, session)
  return { session, loadMs }
}

function argmax(a: ArrayLike<number>, off: number, n: number) {
  let best = 0
  for (let i = 1; i < n; i++) if (a[off + i] > a[off + best]) best = i
  return best
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0))

export async function benchmarkModel(
  model: ModelMeta,
  samples: Fixture[],
  inputs: Float32Array[],
  nClasses: number,
  settings: BenchSettings,
  onProgress?: (done: number, total: number) => void,
): Promise<ModelResult> {
  let backendUsed: string = settings.backend
  let sess: { session: ort.InferenceSession; loadMs: number }
  try {
    sess = await getSession(model, backendUsed)
  } catch {
    // Tree ensembles and integer MatMul are Wasm-only kernels; fall back honestly.
    backendUsed = 'wasm'
    sess = await getSession(model, backendUsed)
  }
  const { session, loadMs } = sess
  const inName = session.inputNames[0]
  const probName = session.outputNames.find((n) => n.includes('prob')) ?? session.outputNames[session.outputNames.length - 1]
  const len = inputs[0].length

  const feed = (x: Float32Array) => ({ [inName]: new ort.Tensor('float32', x, [1, len]) })

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
  const predictions = new Array<number>(samples.length).fill(-1)
  const confidences = new Array<number>(samples.length).fill(0)
  const total = settings.runs * samples.length
  let done = 0
  for (let r = 0; r < settings.runs; r++) {
    for (let i = 0; i < samples.length; i++) {
      const f = feed(inputs[i])
      let out!: ort.InferenceSession.OnnxValueMapType
      const t0 = performance.now()
      for (let k = 0; k < inner; k++) out = await session.run(f, [probName])
      const dt = (performance.now() - t0) / inner
      latencies.push(dt)
      const probs = out[probName].data as Float32Array
      const k = argmax(probs, 0, nClasses)
      predictions[i] = k
      confidences[i] = probs[k]
      done++
    }
    onProgress?.(done, total)
    await tick() // keep the UI responsive between passes
  }

  const correct = predictions.reduce((a, p, i) => a + (p === samples[i].labelIndex ? 1 : 0), 0)
  return {
    modelId: model.id,
    backendUsed,
    latencies,
    predictions,
    confidences,
    correct,
    total: samples.length,
    accuracy: correct / samples.length,
    loadMs,
    callsPerTiming: inner,
    ...summarize(latencies),
  }
}
