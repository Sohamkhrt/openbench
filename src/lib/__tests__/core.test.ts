import { describe, expect, it, vi } from 'vitest'

// bench.ts imports onnxruntime-web, which needs a browser; the pure helpers under test do not.
vi.mock('onnxruntime-web', () => ({ env: { wasm: {} }, InferenceSession: {}, Tensor: class {} }))

import { evs, macroF1, prepareInputs, softmax, summarize } from '../bench'
import { METRICS, SECTIONS, makeRow } from '../metrics'
import { N, PRESETS, resampleMotion, synthesize } from '../synth'
import type { ModelMeta, ModelResult } from '../types'

describe('evs', () => {
  it('matches the formula (accuracy x 100) / (latency ms x size MB)', () => {
    expect(evs(0.9, 2, 0.5)).toBeCloseTo(90)
  })
  it('doubles when latency or size halves', () => {
    expect(evs(0.8, 1, 1)).toBeCloseTo(2 * evs(0.8, 2, 1))
    expect(evs(0.8, 1, 0.5)).toBeCloseTo(2 * evs(0.8, 1, 1))
  })
  it('never divides by zero', () => {
    expect(Number.isFinite(evs(1, 0, 0))).toBe(true)
  })
})

describe('summarize', () => {
  it('computes mean, percentiles, max and std', () => {
    const s = summarize(Array.from({ length: 100 }, (_, i) => i + 1))
    expect(s.meanMs).toBeCloseTo(50.5)
    expect(s.p50Ms).toBe(50)
    expect(s.p95Ms).toBe(95)
    expect(s.maxMs).toBe(100)
    expect(s.stdMs).toBeGreaterThan(28)
  })
  it('handles an empty list', () => {
    expect(summarize([]).maxMs).toBe(0)
  })
})

describe('classification helpers', () => {
  it('softmax sums to one and keeps the order', () => {
    const p = softmax([1, 3, 2, 0])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(p.indexOf(Math.max(...p))).toBe(1)
  })
  it('macro F1 is 1 for a perfect matrix and lower with mistakes', () => {
    expect(macroF1([[5, 0], [0, 5]])).toBe(1)
    expect(macroF1([[4, 1], [2, 3]])).toBeLessThan(1)
  })
})

describe('prepareInputs', () => {
  const fx = [{ id: 'a', label: 'smooth', labelIndex: 0, data: [1, 2, 3, 4] }]
  it('returns the data unchanged without noise', () => {
    expect(Array.from(prepareInputs(fx, 0)[0])).toEqual([1, 2, 3, 4])
  })
  it('adds reproducible noise', () => {
    const a = prepareInputs(fx, 0.1)[0]
    const b = prepareInputs(fx, 0.1)[0]
    expect(Array.from(a)).toEqual(Array.from(b))
    expect(a[0]).not.toBe(1)
  })
})

describe('synthesize', () => {
  it('produces a 3 x 128 window with gravity on z', () => {
    const w = synthesize({ ...PRESETS.smooth, seed: 1 })
    expect(w.length).toBe(3 * N)
    const meanZ = w.slice(2 * N).reduce((a, b) => a + b, 0) / N
    expect(meanZ).toBeGreaterThan(0.9)
    expect(meanZ).toBeLessThan(1.1)
  })
  it('is deterministic for a seed', () => {
    expect(Array.from(synthesize({ ...PRESETS.rough, seed: 3 }))).toEqual(Array.from(synthesize({ ...PRESETS.rough, seed: 3 })))
  })
  it('a pothole drops z well below 1 g and a bump lifts it', () => {
    const z = (w: Float32Array) => Array.from(w.slice(2 * N))
    expect(Math.min(...z(synthesize({ ...PRESETS.pothole, seed: 1 })))).toBeLessThan(0.7)
    expect(Math.max(...z(synthesize({ ...PRESETS.speed_bump, seed: 1 })))).toBeGreaterThan(1.25)
  })
})

describe('resampleMotion', () => {
  it('resamples 60 Hz m/s² samples to 128 points in g with gravity up', () => {
    const samples = Array.from({ length: 90 }, (_, i) => ({ t: i * 16.67, x: 0, y: 0, z: -9.80665 }))
    const w = resampleMotion(samples)
    expect(w.length).toBe(3 * N)
    expect(w[2 * N + 64]).toBeCloseTo(1, 3) // iOS-style negative gravity is flipped
  })
})

describe('metric catalogue', () => {
  it('every metric belongs to a section and has explanations', () => {
    const ids = new Set(SECTIONS.map((s) => s.id))
    for (const m of METRICS) {
      expect(ids.has(m.section)).toBe(true)
      expect(m.what.length).toBeGreaterThan(10)
      expect(m.why.length).toBeGreaterThan(10)
    }
  })
  it('uploaded models take their quality numbers from the live run', () => {
    const meta = { id: 'custom-1', custom: true, sizeMB: 1, heldOutAccuracy: 0 } as ModelMeta
    const result = { accuracy: 0.75, hazardRecall: 0.5, macroF1: 0.7, p50Ms: 1, total: 20 } as ModelResult
    const row = makeRow(meta, result, 'heldout')
    expect(row.meta.heldOutAccuracy).toBe(0.75)
    expect(row.evs).toBeCloseTo(75)
  })
})
