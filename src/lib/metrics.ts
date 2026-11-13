import { ROBUST_SIGMA, evs } from './bench'
import type { ModelMeta, ModelResult } from './types'

export interface Row {
  meta: ModelMeta
  result: ModelResult
  /** accuracy used for EVS (live fixtures or held-out, per user choice) */
  accuracy: number
  evs: number
}

export function makeRow(meta: ModelMeta, result: ModelResult, source: 'live' | 'heldout'): Row {
  if (meta.custom) {
    // No held-out set exists for an uploaded model, so its quality comes from the live fixtures.
    meta = {
      ...meta,
      heldOutAccuracy: result.accuracy,
      heldOutHazardRecall: result.hazardRecall,
      heldOutMacroF1: result.macroF1 ?? 0,
      heldOutConfusion: result.confusion,
      heldOutSamples: result.total,
    }
  }
  const accuracy = source === 'live' ? result.accuracy : meta.heldOutAccuracy
  return { meta, result, accuracy, evs: evs(accuracy, result.p50Ms, meta.sizeMB) }
}

/* ---------- colours: cool hues for neural nets, warm for trees ---------- */

export const MODEL_COLORS: Record<string, string> = {
  cnn_fp32: '#2dd4bf',
  cnn_int8: '#2dd4bf',
  mlp_fp32: '#60a5fa',
  mlp_int8: '#60a5fa',
  wide_fp32: '#818cf8',
  wide_int8: '#818cf8',
  logreg: '#c084fc',
  dtree: '#facc15',
  rf_sklearn: '#fb923c',
  gboost: '#f87171',
  knn: '#94a3b8',
}
export const colorFor = (id: string) => MODEL_COLORS[id] ?? (id.startsWith('custom') ? '#f472b6' : '#a1a1aa')
/** INT8 models are drawn hatched so quantized variants stand out next to their FP32 twin. */
export const isQuantized = (m: ModelMeta) => m.precision === 'int8'
export const fillFor = (m: ModelMeta) => (isQuantized(m) ? `url(#hatch-${m.id})` : colorFor(m.id))

export const FAMILY_LABEL: Record<string, string> = {
  neural: 'Neural net',
  linear: 'Linear',
  trees: 'Tree ensemble',
  instance: 'Instance-based',
  custom: 'Your model',
}

/* ---------- metric catalogue ---------- */

export type Better = 'higher' | 'lower'

export interface MetricDef {
  id: string
  section: SectionId
  title: string
  short: string
  better: Better
  unit: string
  value: (r: Row) => number
  format: (v: number) => string
  /** what is measured, one sentence */
  what: string
  /** what it means for someone choosing a model */
  why: string
  source: 'live' | 'held-out' | 'file'
  /** draw on a log axis when values span orders of magnitude */
  log?: boolean
}

export type SectionId = 'overall' | 'quality' | 'speed' | 'footprint' | 'robustness'

export const SECTIONS: { id: SectionId; title: string; blurb: string }[] = [
  { id: 'overall', title: 'Overall', blurb: 'One number that balances accuracy, speed and size.' },
  { id: 'quality', title: 'Prediction quality', blurb: 'How often the model is right, and whether it catches the hazards that matter.' },
  { id: 'speed', title: 'Speed', blurb: 'How long one prediction takes on this device, measured live.' },
  { id: 'footprint', title: 'Footprint', blurb: 'How much storage and memory the model needs on the device.' },
  { id: 'robustness', title: 'Robustness', blurb: 'How the model copes with a noisier, cheaper sensor.' },
]

const ms = (v: number) => (v >= 1 ? `${v.toFixed(2)} ms` : `${(v * 1000).toFixed(0)} µs`)
const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const compact = (v: number) =>
  v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : v.toFixed(0)

export const METRICS: MetricDef[] = [
  {
    id: 'evs', log: true, section: 'overall', title: 'Edge Viability Score', short: 'EVS', better: 'higher', unit: '',
    value: (r) => r.evs, format: (v) => Math.round(v).toLocaleString(),
    what: 'Accuracy × 100 divided by (p50 latency in ms × size in MB).',
    why: 'A quick "bang for the buck" rank. It strongly rewards tiny, fast models, so always check accuracy next to it: a model can win EVS and still be too inaccurate to ship.',
    source: 'live',
  },
  {
    id: 'accuracy', section: 'quality', title: 'Accuracy', short: 'Accuracy', better: 'higher', unit: '%',
    value: (r) => r.meta.heldOutAccuracy, format: pct,
    what: `Share of ${'1,000'} unseen road windows classified correctly.`,
    why: 'Out of every 100 stretches of road, this is how many the model labels correctly. 90% means about one wrong call every 13 seconds of driving.',
    source: 'held-out',
  },
  {
    id: 'hazard', section: 'quality', title: 'Hazard recall', short: 'Hazard recall', better: 'higher', unit: '%',
    value: (r) => r.meta.heldOutHazardRecall, format: pct,
    what: 'Of all real potholes and speed bumps, the share the model flags correctly.',
    why: 'This is the safety number. A missed pothole is worse than a false alarm on smooth road, so pick the highest hazard recall you can afford.',
    source: 'held-out',
  },
  {
    id: 'f1', section: 'quality', title: 'Macro F1', short: 'Macro F1', better: 'higher', unit: '',
    value: (r) => r.meta.heldOutMacroF1, format: (v) => v.toFixed(3),
    what: 'Average of precision and recall balance across all four road classes.',
    why: 'Catches models that look accurate only because they are good at the common class. Close to accuracy means the model is balanced.',
    source: 'held-out',
  },
  {
    id: 'live', section: 'quality', title: 'Live fixture accuracy', short: 'Fixture acc.', better: 'higher', unit: '%',
    value: (r) => r.result.accuracy, format: pct,
    what: 'Correct answers on the 20 fixtures you just watched, computed in this browser.',
    why: 'Proof the exported ONNX file behaves like the trained model. Only 20 samples, so treat small differences as noise.',
    source: 'live',
  },
  {
    id: 'p50', log: true, section: 'speed', title: 'Typical latency (p50)', short: 'p50', better: 'lower', unit: 'ms',
    value: (r) => r.result.p50Ms, format: ms,
    what: 'Median time for one session.run() call on this device.',
    why: 'How long the device waits for an answer on a normal reading. At 100 Hz sampling you get a new window every 1.28 s, so anything far below that keeps up easily.',
    source: 'live',
  },
  {
    id: 'p95', log: true, section: 'speed', title: 'Worst-case latency (p95)', short: 'p95', better: 'lower', unit: 'ms',
    value: (r) => r.result.p95Ms, format: ms,
    what: '95% of predictions finish faster than this.',
    why: 'Real-time loops break on the slow calls, not the average. Size your timing budget against p95, not p50.',
    source: 'live',
  },
  {
    id: 'jitter', section: 'speed', title: 'Jitter (p95 / p50)', short: 'Jitter', better: 'lower', unit: '×',
    value: (r) => r.result.p95Ms / Math.max(r.result.p50Ms, 1e-6), format: (v) => `${v.toFixed(2)}×`,
    what: 'How much slower the bad calls are than the typical call.',
    why: 'Close to 1× means predictable timing. High jitter means occasional stalls you must design around, for example with a watchdog or a bigger buffer.',
    source: 'live',
  },
  {
    id: 'throughput', log: true, section: 'speed', title: 'Throughput', short: 'Throughput', better: 'higher', unit: '/s',
    value: (r) => 1000 / Math.max(r.result.meanMs, 1e-6), format: (v) => `${compact(v)}/s`,
    what: 'Predictions per second on one thread, from the mean latency.',
    why: 'Headroom. High throughput leaves CPU time for the rest of your firmware, or lets one gateway serve many sensors.',
    source: 'live',
  },
  {
    id: 'size', log: true, section: 'footprint', title: 'Model file size', short: 'Size', better: 'lower', unit: 'MB',
    value: (r) => r.meta.sizeMB, format: (v) => (v < 0.1 ? `${(v * 1024).toFixed(1)} KB` : `${v.toFixed(2)} MB`),
    what: 'Size of the .onnx file the device has to store and load.',
    why: 'Decides which chips it fits on. A microcontroller with 1 MB of flash cannot hold a 4 MB model, and every OTA update has to ship it.',
    source: 'file',
  },
  {
    id: 'params', log: true, section: 'footprint', title: 'Parameters', short: 'Params', better: 'lower', unit: '',
    value: (r) => r.meta.params, format: compact,
    what: 'Learned numbers inside the model: weights for networks, tree nodes for trees, stored samples for kNN.',
    why: 'A rough proxy for RAM and compute. Trees have few nodes but branchy code; networks have many weights but simple maths.',
    source: 'file',
  },
  {
    id: 'cold', section: 'footprint', title: 'Cold start', short: 'Load', better: 'lower', unit: 'ms',
    value: (r) => r.result.loadMs, format: (v) => `${v.toFixed(0)} ms`,
    what: 'Time to download, parse and initialise the model in this browser.',
    why: 'Matters for devices that sleep and wake to save battery: they pay this cost every wake-up.',
    source: 'live',
  },
  {
    id: 'noisy', section: 'robustness', title: `Accuracy with noise (σ = ${ROBUST_SIGMA} g)`, short: 'Noisy acc.', better: 'higher', unit: '%',
    value: (r) => r.result.noisyAccuracy, format: pct,
    what: `Fixture accuracy after adding ${ROBUST_SIGMA} g of random sensor noise to every reading.`,
    why: 'Simulates a cheaper accelerometer or a loose mount. A model that collapses here will struggle on real hardware.',
    source: 'live',
  },
  {
    id: 'drop', section: 'robustness', title: 'Accuracy lost to noise', short: 'Noise drop', better: 'lower', unit: 'pts',
    value: (r) => Math.max(0, r.result.accuracy - r.result.noisyAccuracy) * 100, format: (v) => `${v.toFixed(0)} pts`,
    what: 'Clean fixture accuracy minus noisy fixture accuracy.',
    why: 'Smaller is better. Zero means the model did not notice the extra noise at all.',
    source: 'live',
  },
]

export const metricById = (id: string) => METRICS.find((m) => m.id === id)!

/** Rank rows for one metric, best first. */
export function ranked(rows: Row[], m: MetricDef) {
  const sorted = [...rows].sort((a, b) => (m.better === 'higher' ? m.value(b) - m.value(a) : m.value(a) - m.value(b)))
  return sorted
}

/* ---------- deployment targets ---------- */

export interface DeviceTarget {
  id: string
  name: string
  flashMB: number
  note: string
}

export const DEVICES: DeviceTarget[] = [
  { id: 'nano', name: 'Arduino Nano 33 BLE', flashMB: 0.25, note: 'Cortex-M4, 1 MB flash shared with firmware, ~256 KB left for a model' },
  { id: 'pico', name: 'Raspberry Pi Pico', flashMB: 1, note: 'RP2040, 2 MB flash, ~1 MB realistic for a model' },
  { id: 'esp32', name: 'ESP32-S3', flashMB: 4, note: '8 MB flash, ~4 MB usable after firmware and OTA slot' },
  { id: 'pizero', name: 'Raspberry Pi Zero 2 W', flashMB: 64, note: 'Linux SBC, storage is not a constraint' },
  { id: 'phone', name: 'Smartphone / browser', flashMB: 100, note: 'Plenty of storage; download size still matters' },
]
