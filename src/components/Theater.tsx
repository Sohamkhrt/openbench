import { ArrowRight, Check, Cpu, Radio, Sigma, Waves, X } from 'lucide-react'
import type { Phase } from '../lib/bench'
import { FAMILY_LABEL, colorFor } from '../lib/metrics'
import type { Fixture, ModelMeta } from '../lib/types'

export interface WatchState {
  modelId: string
  sampleIndex: number
  probs: number[]
  ms: number
  /** increments every step so CSS animations restart */
  tick: number
}

const PHASES: { id: Phase; label: string }[] = [
  { id: 'loading', label: 'Load model' },
  { id: 'watching', label: 'Classify 20 windows' },
  { id: 'robustness', label: 'Noise stress test' },
  { id: 'timing', label: 'Precision timing' },
]

export const CLASS_LABEL: Record<string, string> = {
  smooth: 'Smooth road',
  rough: 'Rough road',
  pothole: 'Pothole',
  speed_bump: 'Speed bump',
}

export function Wave({ data, length, sweepMs, tick }: { data: number[]; length: number; sweepMs: number; tick: number }) {
  const W = 600
  const H = 150
  const axes = [
    { off: 0, color: '#a78bfa', name: 'x' },
    { off: length, color: '#f472b6', name: 'y' },
    { off: 2 * length, color: '#22d3ee', name: 'z' },
  ]
  let lo = Infinity
  let hi = -Infinity
  for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v }
  const pad = (hi - lo) * 0.08 || 0.1
  lo -= pad; hi += pad
  const y = (v: number) => H - ((v - lo) / (hi - lo)) * H
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full" preserveAspectRatio="none" role="img" aria-label="3-axis accelerometer window">
      {[0.25, 0.5, 0.75].map((f) => <line key={f} x1={0} x2={W} y1={H * f} y2={H * f} stroke="#27272a" strokeDasharray="3 4" />)}
      {axes.map((a) => (
        <polyline
          key={a.name}
          fill="none"
          stroke={a.color}
          strokeWidth={a.name === 'z' ? 1.8 : 1.1}
          strokeOpacity={a.name === 'z' ? 1 : 0.75}
          vectorEffect="non-scaling-stroke"
          points={Array.from({ length }, (_, i) => `${(i / (length - 1)) * W},${y(data[a.off + i])}`).join(' ')}
        />
      ))}
      {sweepMs > 0 && (
        <g key={tick} className="sweep" style={{ animationDuration: `${sweepMs}ms` }}>
          <rect x={-40} y={0} width={40} height={H} fill="url(#sweepGrad)" />
          <line x1={0} x2={0} y1={0} y2={H} stroke="#34d399" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </g>
      )}
      <defs>
        <linearGradient id="sweepGrad" x1="0" x2="1">
          <stop offset="0" stopColor="#34d399" stopOpacity="0" />
          <stop offset="1" stopColor="#34d399" stopOpacity="0.25" />
        </linearGradient>
      </defs>
    </svg>
  )
}

export function Theater(props: {
  models: ModelMeta[]
  classes: string[]
  samples: Fixture[]
  inputs: Float32Array[]
  length: number
  watch: WatchState | null
  phases: Record<string, Phase>
  cells: Record<string, (boolean | null)[]>
  progress: Record<string, number>
  running: boolean
  sweepMs: number
  onPickSample: (i: number) => void
  selected: number
}) {
  const { watch, models, classes, samples } = props
  const active = watch ? models.find((m) => m.id === watch.modelId) : null
  const sIdx = watch?.sampleIndex ?? props.selected
  const sample = samples[sIdx]
  const data = props.inputs[sIdx] ? Array.from(props.inputs[sIdx]) : sample.data
  const pred = watch ? watch.probs.indexOf(Math.max(...watch.probs)) : -1
  const correct = pred === sample.labelIndex
  const phase = active ? props.phases[active.id] : undefined
  const color = active ? colorFor(active.id) : '#52525b'

  return (
    <section id="live" className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${props.running ? 'animate-pulse bg-emerald-400' : 'bg-zinc-600'}`} />
          <h2 className="text-sm font-medium text-zinc-100">Live inference</h2>
          <span className="text-xs text-zinc-500">
            {props.running && active ? <>Running <b style={{ color }}>{active.name}</b> in your browser</> : 'Watch each model read the sensor and decide what road it is on'}
          </span>
        </div>
        {/* phase stepper */}
        <ol className="flex flex-wrap items-center gap-1 text-[11px]">
          {PHASES.map((p, i) => {
            const order = PHASES.findIndex((x) => x.id === phase)
            const state = !phase ? 'idle' : phase === 'done' || i < order ? 'done' : i === order ? 'active' : 'idle'
            return (
              <li key={p.id} className="flex items-center gap-1">
                <span className={`rounded-full border px-2 py-0.5 transition ${state === 'active' ? 'border-emerald-400/50 bg-emerald-400/10 text-emerald-300' : state === 'done' ? 'border-zinc-700 text-zinc-400' : 'border-zinc-800 text-zinc-600'}`}>
                  {state === 'done' && <Check className="mr-0.5 inline h-3 w-3" />}{p.label}
                </span>
                {i < PHASES.length - 1 && <span className="text-zinc-700">›</span>}
              </li>
            )
          })}
        </ol>
      </div>

      <div className="grid gap-4 p-4 lg:grid-cols-[1fr_260px]">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <span className="font-mono text-zinc-400">{sample.id}</span>
            <span className="text-zinc-600">·</span>
            <span className="text-zinc-500">1.28 s of driving, 3-axis accelerometer @ 100 Hz</span>
            <span className="ml-auto flex gap-3 font-mono text-[11px]">
              <span className="text-violet-400">x</span><span className="text-pink-400">y</span><span className="text-cyan-400">z (vertical)</span>
            </span>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            <Wave data={data} length={props.length} sweepMs={props.running ? props.sweepMs : 0} tick={watch?.tick ?? 0} />
          </div>

          {/* pipeline */}
          <div key={watch?.tick} className="mt-3 grid grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] items-center gap-1.5 text-[11px]">
            <Stage n={0} on={!!watch} icon={<Radio className="h-3.5 w-3.5" />} title="Sensor window" sub="3 × 128 readings" delay={props.sweepMs} />
            <ArrowRight className="h-3.5 w-3.5 text-zinc-700" />
            <Stage n={1} on={!!watch} icon={<Waves className="h-3.5 w-3.5" />} title="Tensor" sub="[1, 384] float32" delay={props.sweepMs} />
            <ArrowRight className="h-3.5 w-3.5 text-zinc-700" />
            <Stage n={2} on={!!watch} icon={<Cpu className="h-3.5 w-3.5" />} title={active?.name ?? 'Model'} sub={active ? FAMILY_LABEL[active.family] : 'onnxruntime-web'} delay={props.sweepMs} color={color} />
            <ArrowRight className="h-3.5 w-3.5 text-zinc-700" />
            <Stage n={3} on={!!watch} icon={<Sigma className="h-3.5 w-3.5" />} title="Probabilities" sub={watch ? `${(watch.ms * 1000).toFixed(0)} µs` : '4 classes'} delay={props.sweepMs} />
          </div>
        </div>

        {/* decision */}
        <div className="flex flex-col gap-2">
          <p className="label">Model's answer</p>
          {classes.map((c, i) => {
            const p = watch ? watch.probs[i] ?? 0 : 0
            const isPred = i === pred
            return (
              <div key={c}>
                <div className="mb-0.5 flex justify-between text-xs">
                  <span className={isPred ? 'text-zinc-100' : 'text-zinc-500'}>{CLASS_LABEL[c] ?? c}{i === sample.labelIndex && <span className="ml-1 text-[10px] text-zinc-500">(truth)</span>}</span>
                  <span className="font-mono text-zinc-400">{(p * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
                  <div className="h-full rounded-full" style={{ width: `${p * 100}%`, background: isPred ? color : '#3f3f46' }} />
                </div>
              </div>
            )
          })}
          <div className={`mt-1 flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${!watch ? 'border-zinc-800 text-zinc-500' : correct ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' : 'border-rose-500/40 bg-rose-500/10 text-rose-300'}`}>
            {!watch ? 'Press Run to start' : correct ? <><Check className="h-4 w-4" /> Correct: {CLASS_LABEL[classes[pred]]}</> : <><X className="h-4 w-4" /> Wrong: said {CLASS_LABEL[classes[pred]]}</>}
          </div>
        </div>
      </div>

      {/* scoreboard */}
      <div className="border-t border-zinc-800 px-4 py-3">
        <div className="mb-2 flex items-center justify-between">
          <p className="label">Scoreboard · every model × every road window</p>
          <p className="flex items-center gap-3 text-[11px] text-zinc-500">
            <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-emerald-500/80" />right</span>
            <span className="flex items-center gap-1"><i className="h-2.5 w-2.5 rounded-sm bg-rose-500/80" />wrong</span>
            <span className="hidden sm:inline">click a column to inspect it</span>
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[620px] border-separate border-spacing-[3px] text-[11px]">
            <thead>
              <tr>
                <th />
                {samples.map((s, i) => (
                  <th key={s.id} className="p-0">
                    <button onClick={() => props.onPickSample(i)} className={`w-full rounded px-0.5 font-mono font-normal ${i === sIdx ? 'text-zinc-100' : 'text-zinc-600 hover:text-zinc-300'}`} title={s.id}>
                      {s.label === 'speed_bump' ? 'B' : s.label[0].toUpperCase()}
                    </button>
                  </th>
                ))}
                <th className="pl-2 text-right font-normal text-zinc-600">score</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const row = props.cells[m.id] ?? []
                const right = row.filter((c) => c === true).length
                const seen = row.filter((c) => c !== null && c !== undefined).length
                const isActive = watch?.modelId === m.id && props.running
                return (
                  <tr key={m.id}>
                    <td className="whitespace-nowrap pr-2">
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: colorFor(m.id) }} />
                        <span className={isActive ? 'text-zinc-100' : 'text-zinc-400'}>{m.name}</span>
                        {props.phases[m.id] === 'timing' && <span className="font-mono text-zinc-600">{props.progress[m.id] ?? 0}%</span>}
                      </span>
                    </td>
                    {samples.map((s, i) => {
                      const c = row[i]
                      const now = isActive && watch?.sampleIndex === i
                      return (
                        <td key={s.id} className="p-0">
                          <div className={`h-4 rounded-[3px] transition-colors duration-200 ${now ? 'ring-2 ring-white/70' : ''} ${c === true ? 'bg-emerald-500/80' : c === false ? 'bg-rose-500/80' : 'bg-zinc-800/80'}`} />
                        </td>
                      )
                    })}
                    <td className="pl-2 text-right font-mono text-zinc-400">{seen ? `${right}/${seen}` : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}

function Stage({ n, on, icon, title, sub, delay, color }: { n: number; on: boolean; icon: React.ReactNode; title: string; sub: string; delay: number; color?: string }) {
  return (
    <div
      className={`stage min-w-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2 py-1.5 ${on ? 'stage-on' : ''}`}
      style={{ animationDelay: `${(delay * n) / 4}ms`, ['--stage-color' as string]: color ?? '#34d399' }}
    >
      <div className="flex items-center gap-1 text-zinc-300">{icon}<span className="truncate font-medium">{title}</span></div>
      <div className="truncate font-mono text-[10px] text-zinc-500">{sub}</div>
    </div>
  )
}
