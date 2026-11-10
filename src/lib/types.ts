export interface ModelMeta {
  id: string
  name: string
  description: string
  precision: 'fp32' | 'int8' | 'trees'
  file: string
  sizeMB: number
  heldOutAccuracy: number
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
}

export interface ModelResult {
  modelId: string
  backendUsed: string
  /** one entry per timed inference, in order */
  latencies: number[]
  /** predicted class index per fixture (last timed run) */
  predictions: number[]
  confidences: number[]
  correct: number
  total: number
  accuracy: number
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
