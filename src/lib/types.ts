export type Family = 'neural' | 'linear' | 'trees' | 'instance'

export interface ModelMeta {
  id: string
  name: string
  family: Family
  description: string
  precision: 'fp32' | 'int8' | 'trees'
  file: string
  sizeMB: number
  params: number
  heldOutAccuracy: number
  heldOutMacroF1: number
  heldOutHazardRecall: number
  heldOutRecall: number[]
  heldOutSamples: number
}

export interface Manifest {
  classes: string[]
  input: { channels: number; length: number; fs: number; layout: string }
  models: ModelMeta[]
}

export interface Fixture {
  id: string
  label: string
  labelIndex: number
  data: number[]
}

export interface FixtureSet {
  name: string
  fs: number
  length: number
  classes: string[]
  samples: Fixture[]
}

export type Backend = 'wasm' | 'webgpu'

export interface BenchSettings {
  backend: Backend
  runs: number
  warmup: number
  noise: number
  /** ms pause per sample in the visual pass; 0 skips the animation */
  watchDelay: number
}

export interface ModelResult {
  modelId: string
  backendUsed: string
  /** one entry per timed inference, in order */
  latencies: number[]
  /** predicted class index per fixture */
  predictions: number[]
  confidences: number[]
  /** full class probabilities per fixture */
  probs: number[][]
  correct: number
  total: number
  accuracy: number
  /** share of pothole + speed bump fixtures classified correctly */
  hazardRecall: number
  /** fixture accuracy with ROBUST_SIGMA sensor noise added */
  noisyAccuracy: number
  meanMs: number
  p50Ms: number
  p95Ms: number
  maxMs: number
  stdMs: number
  loadMs: number
  /** back-to-back session.run() calls averaged into each latency sample */
  callsPerTiming?: number
  error?: string
}

export interface BenchReport {
  createdAt: string
  userAgent: string
  settings: BenchSettings
  results: ModelResult[]
}
