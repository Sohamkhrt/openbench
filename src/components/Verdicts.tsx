import { AlertTriangle, CheckCircle2, Sparkles, XCircle } from 'lucide-react'
import { DEVICES, type DeviceTarget, type Row, colorFor, FAMILY_LABEL } from '../lib/metrics'

export interface Constraints {
  deviceId: string
  minAccuracy: number
  latencyBudgetMs: number
}

type Status = 'ship' | 'inaccurate' | 'too-big' | 'too-slow'

function status(r: Row, dev: DeviceTarget, c: Constraints): Status {
  if (r.meta.sizeMB > dev.flashMB) return 'too-big'
  if (r.meta.heldOutAccuracy < c.minAccuracy) return 'inaccurate'
  if (r.result.p95Ms > c.latencyBudgetMs) return 'too-slow'
  return 'ship'
}

const STATUS_UI: Record<Status, { label: string; cls: string; icon: React.ReactNode }> = {
  ship: { label: 'Good to ship', cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300', icon: <CheckCircle2 className="h-3.5 w-3.5" /> },
  inaccurate: { label: 'Below your accuracy bar', cls: 'border-amber-500/40 bg-amber-500/10 text-amber-300', icon: <AlertTriangle className="h-3.5 w-3.5" /> },
  'too-big': { label: 'Too big for this device', cls: 'border-rose-500/40 bg-rose-500/10 text-rose-300', icon: <XCircle className="h-3.5 w-3.5" /> },
  'too-slow': { label: 'Over latency budget', cls: 'border-rose-500/40 bg-rose-500/10 text-rose-300', icon: <XCircle className="h-3.5 w-3.5" /> },
}

const us = (ms: number) => (ms >= 1 ? `${ms.toFixed(2)} ms` : `${Math.round(ms * 1000)} µs`)
const size = (mb: number) => (mb < 0.1 ? `${(mb * 1024).toFixed(0)} KB` : `${mb.toFixed(2)} MB`)

/** Score used to pick a recommendation among models that satisfy every constraint. */
const pickScore = (r: Row) => r.meta.heldOutHazardRecall * 2 + r.meta.heldOutAccuracy - r.meta.sizeMB * 0.02

function sentences(r: Row, rows: Row[], dev: DeviceTarget): string[] {
  const m = r.meta
  const n = rows.length
  const rank = (val: (x: Row) => number, higher: boolean) =>
    1 + rows.filter((x) => (higher ? val(x) > val(r) : val(x) < val(r))).length
  const out: string[] = []
  const missEvery = 1 / Math.max(1e-6, 1 - m.heldOutHazardRecall)
  out.push(
    `Gets about ${Math.round(m.heldOutAccuracy * 100)} of every 100 road windows right and flags ${Math.round(m.heldOutHazardRecall * 100)}% of potholes and speed bumps` +
      (m.heldOutHazardRecall < 0.999 ? `, so roughly 1 hazard in ${Math.round(missEvery)} slips through.` : '.'),
  )
  const sp = rank((x) => x.result.p50Ms, false)
  out.push(`Answers in ${us(r.result.p50Ms)} typically and ${us(r.result.p95Ms)} on slow calls, #${sp} of ${n} for speed on this device.`)
  out.push(
    m.sizeMB <= dev.flashMB
      ? `At ${size(m.sizeMB)} it fits on a ${dev.name} (budget ${size(dev.flashMB)}).`
      : `At ${size(m.sizeMB)} it does not fit a ${dev.name} (budget ${size(dev.flashMB)}); you would need a bigger chip.`,
  )
  const drop = Math.max(0, r.result.accuracy - r.result.noisyAccuracy) * 100
  out.push(drop < 1 ? 'Shrugged off the extra sensor noise in the stress test.' : `Lost ${drop.toFixed(0)} accuracy points when the sensor got noisier.`)
  if (m.precision === 'int8') {
    const twin = rows.find((x) => x.meta.id === m.id.replace('_int8', '_fp32'))
    if (twin) {
      const d = (m.heldOutAccuracy - twin.meta.heldOutAccuracy) * 100
      out.push(`INT8 quantization made it ${(twin.meta.sizeMB / m.sizeMB).toFixed(1)}× smaller than ${twin.meta.name} for ${d >= 0 ? `no` : `${Math.abs(d).toFixed(1)} pts of`} accuracy loss.`)
    }
  }
  if (m.family === 'instance') out.push('kNN compares every new window against all stored training windows, so cost grows with your dataset. Rarely a good edge choice.')
  return out
}

export function Recommendations({ rows, c, setC }: { rows: Row[]; c: Constraints; setC: (c: Constraints) => void }) {
  const dev = DEVICES.find((d) => d.id === c.deviceId) ?? DEVICES[0]
  const ok = rows.filter((r) => status(r, dev, c) === 'ship').sort((a, b) => pickScore(b) - pickScore(a))
  const pick = ok[0]
  const runnerUp = ok[1]
  const sorted = [...rows].sort((a, b) => {
    const order: Status[] = ['ship', 'inaccurate', 'too-slow', 'too-big']
    const d = order.indexOf(status(a, dev, c)) - order.indexOf(status(b, dev, c))
    return d || pickScore(b) - pickScore(a)
  })

  return (
    <div className="space-y-4">
      {/* constraints */}
      <div className="card grid gap-4 p-4 md:grid-cols-3">
        <label className="block text-xs">
          <span className="label">Target device</span>
          <select value={c.deviceId} onChange={(e) => setC({ ...c, deviceId: e.target.value })} className="mt-1.5 w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
            {DEVICES.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <span className="mt-1 block text-[11px] text-zinc-500">{dev.note}</span>
        </label>
        <label className="block text-xs">
          <div className="flex justify-between"><span className="label">Minimum accuracy</span><span className="font-mono text-zinc-200">{Math.round(c.minAccuracy * 100)}%</span></div>
          <input type="range" min={0.5} max={0.99} step={0.01} value={c.minAccuracy} onChange={(e) => setC({ ...c, minAccuracy: +e.target.value })} className="mt-2" />
          <span className="mt-1 block text-[11px] text-zinc-500">Held-out accuracy the model must reach.</span>
        </label>
        <label className="block text-xs">
          <div className="flex justify-between"><span className="label">Latency budget (p95)</span><span className="font-mono text-zinc-200">{us(c.latencyBudgetMs)}</span></div>
          <input type="range" min={0.01} max={2} step={0.01} value={c.latencyBudgetMs} onChange={(e) => setC({ ...c, latencyBudgetMs: +e.target.value })} className="mt-2" />
          <span className="mt-1 block text-[11px] text-zinc-500">Measured in this browser; a microcontroller is roughly 10 to 100× slower.</span>
        </label>
      </div>

      {/* headline pick */}
      <div className="card relative overflow-hidden p-5">
        <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full opacity-20 blur-3xl" style={{ background: pick ? colorFor(pick.meta.id) : '#f43f5e' }} />
        <p className="label flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5 text-emerald-400" /> Our pick for a {dev.name}</p>
        {pick ? (
          <>
            <h3 className="mt-2 text-xl font-semibold text-zinc-50">
              <span className="mr-2 inline-block h-3 w-3 rounded-full align-middle" style={{ background: colorFor(pick.meta.id) }} />
              {pick.meta.name}
            </h3>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-zinc-300">
              It meets every constraint you set and has the best hazard detection among the models that do: {Math.round(pick.meta.heldOutHazardRecall * 100)}% of potholes and bumps caught, {Math.round(pick.meta.heldOutAccuracy * 100)}% accuracy overall, {size(pick.meta.sizeMB)} on disk and {us(pick.result.p50Ms)} per prediction.
              {runnerUp && <> Runner-up: <b className="text-zinc-100">{runnerUp.meta.name}</b> ({Math.round(runnerUp.meta.heldOutAccuracy * 100)}%, {size(runnerUp.meta.sizeMB)}).</>}
            </p>
          </>
        ) : (
          <p className="mt-2 text-sm text-zinc-300">No model meets all three constraints. Lower the accuracy bar, raise the latency budget or choose a bigger device.</p>
        )}
      </div>

      {/* per model */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {sorted.map((r) => {
          const s = status(r, dev, c)
          const ui = STATUS_UI[s]
          return (
            <article key={r.meta.id} className="card relative flex flex-col p-4">
              <div className="absolute inset-x-0 top-0 h-0.5 rounded-t-xl" style={{ background: colorFor(r.meta.id) }} />
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-[11px] text-zinc-500">If you pick</p>
                  <h4 className="text-sm font-semibold text-zinc-100">{r.meta.name}</h4>
                  <p className="text-[11px] text-zinc-500">{FAMILY_LABEL[r.meta.family]} · {r.meta.description}</p>
                </div>
              </div>
              <span className={`mt-3 flex items-center gap-1 self-start rounded-full border px-2 py-0.5 text-[11px] ${ui.cls}`}>{ui.icon}{ui.label}</span>
              <ul className="mt-3 space-y-1.5 text-xs leading-relaxed text-zinc-300">
                {sentences(r, rows, dev).map((t, i) => <li key={i} className="flex gap-2"><span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-zinc-600" />{t}</li>)}
              </ul>
            </article>
          )
        })}
      </div>
    </div>
  )
}
