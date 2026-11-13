import { useEffect, useMemo, useRef, useState } from 'react'
import { Shuffle, Smartphone } from 'lucide-react'
import { predict } from '../lib/bench'
import { colorFor } from '../lib/metrics'
import { PRESETS, type MotionSample, type RoadParams, resampleMotion, synthesize } from '../lib/synth'
import type { ModelMeta } from '../lib/types'
import { CLASS_LABEL, Wave } from './Theater'

type Prediction = { probs: number[]; k: number } | { error: string }

interface MotionEventCtor {
  requestPermission?: () => Promise<'granted' | 'denied'>
}

/** Record ~1.3 s from the device accelerometer. Rejects if the device sends no data. */
function recordMotion(): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const samples: MotionSample[] = []
    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity
      if (a && a.x != null && a.y != null && a.z != null) samples.push({ t: performance.now(), x: a.x, y: a.y, z: a.z })
    }
    window.addEventListener('devicemotion', onMotion)
    setTimeout(() => {
      window.removeEventListener('devicemotion', onMotion)
      if (samples.length < 20) reject(new Error('No motion data received. Open this page on a phone and try again.'))
      else resolve(resampleMotion(samples))
    }, 1400)
  })
}

export function Lab({ models, classes, disabled }: { models: ModelMeta[]; classes: string[]; disabled: boolean }) {
  const [params, setParams] = useState<RoadParams>({ ...PRESETS.pothole, seed: 7 })
  const [recorded, setRecorded] = useState<Float32Array | null>(null)
  const [preds, setPreds] = useState<Record<string, Prediction>>({})
  const [active, setActive] = useState(false)
  const [phone, setPhone] = useState<{ state: 'idle' | 'recording' | 'error'; msg?: string }>({ state: 'idle' })
  const runId = useRef(0)

  const signal = useMemo(() => recorded ?? synthesize(params), [params, recorded])

  // Re-run every enabled model whenever the signal changes. Starts on first interaction
  // so the page does not load 11 models before anyone asks for them.
  useEffect(() => {
    if (!active || disabled) return
    const id = ++runId.current
    const timer = setTimeout(async () => {
      for (const m of models) {
        let p: Prediction
        try {
          const probs = await predict(m, signal)
          p = { probs, k: probs.indexOf(Math.max(...probs)) }
        } catch (e) {
          p = { error: e instanceof Error ? e.message : String(e) }
        }
        if (runId.current !== id) return
        setPreds((prev) => ({ ...prev, [m.id]: p }))
      }
    }, 60)
    return () => clearTimeout(timer)
  }, [signal, models, active, disabled])

  const set = (patch: Partial<RoadParams>) => { setRecorded(null); setActive(true); setParams((p) => ({ ...p, ...patch })) }

  async function usePhone() {
    try {
      const ctor = (window as unknown as { DeviceMotionEvent?: MotionEventCtor }).DeviceMotionEvent
      if (!ctor) throw new Error('This browser has no motion sensor API.')
      if (ctor.requestPermission && (await ctor.requestPermission()) !== 'granted') throw new Error('Motion sensor permission was denied.')
      setPhone({ state: 'recording' })
      const data = await recordMotion()
      setRecorded(data)
      setActive(true)
      setPhone({ state: 'idle' })
    } catch (e) {
      setPhone({ state: 'error', msg: e instanceof Error ? e.message : String(e) })
    }
  }

  const votes = classes.map((_, i) => models.filter((m) => { const p = preds[m.id]; return p && 'k' in p && p.k === i }).length)
  const voted = votes.reduce((a, b) => a + b, 0)
  const top = votes.indexOf(Math.max(...votes))

  return (
    <div className="card grid gap-5 p-4 lg:grid-cols-[280px_1fr]">
      <div className="space-y-4">
        <div>
          <p className="label mb-1.5">Start from</p>
          <div className="grid grid-cols-2 gap-1.5 text-xs">
            {Object.keys(PRESETS).map((k) => (
              <button key={k} onClick={() => set(PRESETS[k])} className="rounded-md border border-zinc-800 px-2 py-1.5 text-zinc-300 transition hover:border-zinc-600 hover:text-zinc-50">
                {CLASS_LABEL[k]}
              </button>
            ))}
          </div>
        </div>
        <Knob label="Road roughness" v={params.roughness} min={0} max={0.25} step={0.01} unit="g" onChange={(v) => set({ roughness: v })} />
        <Knob label="Pothole depth" v={params.pothole} min={0} max={0.9} step={0.02} unit="g" onChange={(v) => set({ pothole: v })} />
        <Knob label="Speed-bump height" v={params.bump} min={0} max={0.6} step={0.02} unit="g" onChange={(v) => set({ bump: v })} />
        <Knob label="Vehicle speed" v={params.speed} min={15} max={60} step={1} unit="km/h" digits={0} onChange={(v) => set({ speed: v })} />
        <Knob label="Sensor noise" v={params.noise} min={0} max={0.3} step={0.01} unit="g" onChange={(v) => set({ noise: v })} />
        <div className="flex gap-2 text-xs">
          <button onClick={() => set({ seed: params.seed + 1 })} className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-zinc-800 py-1.5 text-zinc-300 hover:border-zinc-600">
            <Shuffle className="h-3.5 w-3.5" /> New road
          </button>
          <button onClick={usePhone} disabled={phone.state === 'recording'} className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-zinc-800 py-1.5 text-zinc-300 hover:border-zinc-600 disabled:opacity-60">
            <Smartphone className="h-3.5 w-3.5" /> {phone.state === 'recording' ? 'Recording…' : 'Use my phone'}
          </button>
        </div>
        <p className="text-[11px] leading-relaxed text-zinc-500">
          {phone.state === 'error' ? <span className="text-amber-300">{phone.msg}</span> : 'On a phone, "Use my phone" records 1.28 s from its accelerometer. Lay it flat, then tap or bump the table.'}
        </p>
      </div>

      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <span>{recorded ? 'Recorded from your device' : 'Synthesized in your browser'} · 1.28 s @ 100 Hz</span>
          <span className="ml-auto flex gap-3 font-mono text-[11px]">
            <span className="text-violet-400">x</span><span className="text-pink-400">y</span><span className="text-cyan-400">z (vertical)</span>
          </span>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
          <Wave data={Array.from(signal)} length={128} sweepMs={0} tick={0} />
        </div>

        {!active ? (
          <button onClick={() => setActive(true)} disabled={disabled} className="mt-3 w-full rounded-lg border border-dashed border-zinc-700 py-6 text-sm text-zinc-400 transition hover:border-emerald-500/50 hover:text-emerald-300 disabled:opacity-50">
            Move a slider, or click here, to ask all {models.length} models what road this is
          </button>
        ) : (
          <>
            <p className="mt-3 text-sm text-zinc-300">
              {voted === 0 ? 'Asking the models…' : <><b className="text-zinc-50">{votes[top]} of {voted}</b> models say <b className="text-zinc-50">{CLASS_LABEL[classes[top]]}</b></>}
            </p>
            <ul className="mt-2 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
              {models.map((m) => {
                const p = preds[m.id]
                return (
                  <li key={m.id} className="grid grid-cols-[minmax(0,9rem)_1fr_auto] items-center gap-2 text-xs">
                    <span className="flex items-center gap-1.5 truncate text-zinc-400"><i className="h-2 w-2 shrink-0 rounded-full" style={{ background: colorFor(m.id) }} />{m.name}</span>
                    <span className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                      {p && 'k' in p && <span className="block h-full rounded-full" style={{ width: `${p.probs[p.k] * 100}%`, background: colorFor(m.id) }} />}
                    </span>
                    <span className={`w-28 text-right font-mono ${p && 'k' in p && p.k !== top ? 'text-amber-300' : 'text-zinc-300'}`}>
                      {!p ? '…' : 'error' in p ? 'error' : `${CLASS_LABEL[classes[p.k]]} ${(p.probs[p.k] * 100).toFixed(0)}%`}
                    </span>
                  </li>
                )
              })}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

function Knob({ label, v, min, max, step, unit, digits = 2, onChange }: { label: string; v: number; min: number; max: number; step: number; unit: string; digits?: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-zinc-200">{v.toFixed(digits)} {unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}
