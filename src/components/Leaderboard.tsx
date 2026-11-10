import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ChevronDown, Trophy } from 'lucide-react'
import { type MetricDef, type Row, colorFor, fillFor, ranked } from '../lib/metrics'

/** Hatch patterns for quantized models, defined once and referenced by every chart via url(#hatch-id). */
export function GlobalHatchDefs({ ids }: { ids: string[] }) {
  return (
    <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden>
      <defs>
        {ids.map((id) => (
          <pattern key={id} id={`hatch-${id}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill={colorFor(id)} fillOpacity="0.35" />
            <rect width="3" height="6" fill={colorFor(id)} />
          </pattern>
        ))}
      </defs>
    </svg>
  )
}

const SOURCE_LABEL = { live: 'measured live', 'held-out': '1,000 held-out windows', file: 'from the model file' }

export function MetricCard({ metric, rows, highlight }: { metric: MetricDef; rows: Row[]; highlight?: string | null }) {
  const [open, setOpen] = useState(false)
  const data = ranked(rows, metric).map((r) => ({ id: r.meta.id, name: r.meta.name, v: metric.value(r), label: metric.format(metric.value(r)), row: r }))
  const best = data[0]
  return (
    <article className="card flex flex-col p-4">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-[15px] font-semibold text-zinc-100">{metric.title}</h3>
          <p className="mt-0.5 text-[11px] text-zinc-500">
            {metric.better === 'higher' ? 'Higher is better' : 'Lower is better'} · {SOURCE_LABEL[metric.source]}{metric.log ? ' · log scale' : ''}
          </p>
        </div>
        {best && (
          <span className="flex shrink-0 items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300">
            <Trophy className="h-3 w-3" style={{ color: colorFor(best.id) }} /> {best.name}
          </span>
        )}
      </header>
      <p className="mt-2 text-xs leading-relaxed text-zinc-400">{metric.what}</p>

      <div className="mt-2 h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 8, left: 52, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke="#27272a" strokeDasharray="2 4" />
            <XAxis dataKey="name" interval={0} tickLine={false} axisLine={{ stroke: '#3f3f46' }} height={78}
              tick={({ x, y, payload }) => (
                <text x={x} y={y + 8} textAnchor="end" transform={`rotate(-40 ${x} ${y + 8})`} fontSize={10.5} fill={highlight === data.find((d) => d.name === payload.value)?.id ? '#fafafa' : '#a1a1aa'}>
                  {payload.value}
                </text>
              )}
            />
            {metric.log ? (
              <YAxis hide scale="log" domain={[(min: number) => min / 3, (max: number) => max * 1.6]} allowDataOverflow />
            ) : (
              <YAxis hide domain={[0, 'auto']} />
            )}
            <Tooltip
              cursor={{ fill: '#27272a66' }}
              contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#e4e4e7' }}
              formatter={(v: number) => [metric.format(v), metric.short]}
            />
            <Bar dataKey="v" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.id} fill={fillFor(d.row.meta)} fillOpacity={highlight && highlight !== d.id ? 0.35 : 1} />
              ))}
              <LabelList dataKey="label" position="top" fontSize={10} fill="#d4d4d8" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <button onClick={() => setOpen(!open)} className="mt-1 flex items-center gap-1 self-start text-[11px] text-emerald-400/90 hover:text-emerald-300">
        What this means for you <ChevronDown className={`h-3 w-3 transition ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <p className="mt-1.5 rounded-lg bg-zinc-800/50 p-2.5 text-xs leading-relaxed text-zinc-300">{metric.why}</p>}
    </article>
  )
}
