import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import type { ModelMeta, ModelResult } from '../lib/types'

export const MODEL_COLORS: Record<string, string> = {
  mlp_fp32: '#60a5fa', // blue
  mlp_int8: '#34d399', // emerald
  rf_sklearn: '#f59e0b', // amber
}
export const colorFor = (id: string) => MODEL_COLORS[id] ?? '#a1a1aa'

const axis = { stroke: '#52525b', fontSize: 11, tickLine: false, axisLine: { stroke: '#3f3f46' } }
const grid = <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
const tooltip = {
  contentStyle: { background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#a1a1aa' },
  cursor: { fill: '#27272a55' },
}

export interface Row {
  meta: ModelMeta
  result: ModelResult
  accuracy: number
  evs: number
}

export function EvsBar({ rows }: { rows: Row[] }) {
  const data = rows.map((r) => ({ name: r.meta.name, id: r.meta.id, evs: +r.evs.toFixed(1) }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        {grid}
        <XAxis dataKey="name" {...axis} />
        <YAxis {...axis} />
        <Tooltip {...tooltip} formatter={(v: number) => [v.toLocaleString(), 'EVS']} />
        <Bar dataKey="evs" radius={[6, 6, 0, 0]}>
          {data.map((d) => <Cell key={d.id} fill={colorFor(d.id)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function LatencyTimeline({ rows, budget }: { rows: Row[]; budget: number }) {
  const n = Math.max(0, ...rows.map((r) => r.result.latencies.length))
  const data = Array.from({ length: n }, (_, i) => {
    const o: Record<string, number> = { i: i + 1 }
    rows.forEach((r) => { if (r.result.latencies[i] !== undefined) o[r.meta.id] = +r.result.latencies[i].toFixed(3) })
    return o
  })
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        {grid}
        <XAxis dataKey="i" {...axis} minTickGap={32} />
        <YAxis {...axis} unit="ms" />
        <Tooltip {...tooltip} labelFormatter={(l) => `Inference #${l}`} formatter={(v: number, k: string) => [`${v} ms`, rows.find((r) => r.meta.id === k)?.meta.name ?? k]} />
        <ReferenceLine y={budget} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: 'budget', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
        {rows.map((r) => (
          <Line key={r.meta.id} type="monotone" dataKey={r.meta.id} stroke={colorFor(r.meta.id)} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function TradeoffScatter({ rows }: { rows: Row[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <ScatterChart margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
        {grid}
        <XAxis type="number" dataKey="lat" name="p50 latency" unit="ms" {...axis} />
        <YAxis type="number" dataKey="acc" name="accuracy" unit="%" domain={[0, 100]} {...axis} />
        <ZAxis type="number" dataKey="size" range={[120, 900]} name="size" unit="MB" />
        <Tooltip {...tooltip} cursor={{ strokeDasharray: '3 3' }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {rows.map((r) => (
          <Scatter
            key={r.meta.id}
            name={r.meta.name}
            fill={colorFor(r.meta.id)}
            data={[{ lat: +r.result.p50Ms.toFixed(3), acc: +(r.accuracy * 100).toFixed(1), size: r.meta.sizeMB }]}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

export function SignalPlot({ data, length }: { data: number[]; length: number }) {
  const rows = Array.from({ length }, (_, i) => ({
    t: +(i / 100).toFixed(2), x: data[i], y: data[length + i], z: data[2 * length + i],
  }))
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={rows} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        {grid}
        <XAxis dataKey="t" {...axis} unit="s" minTickGap={40} />
        <YAxis {...axis} unit="g" domain={['auto', 'auto']} />
        <Tooltip {...tooltip} formatter={(v: number) => v.toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="x" stroke="#a78bfa" dot={false} strokeWidth={1.2} isAnimationActive={false} />
        <Line dataKey="y" stroke="#f472b6" dot={false} strokeWidth={1.2} isAnimationActive={false} />
        <Line dataKey="z" stroke="#22d3ee" dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
