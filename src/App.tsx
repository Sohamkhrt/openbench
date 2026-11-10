import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, CheckCircle2, Cpu, Download, Gauge, Github, HardDrive, Info, Play,
  RotateCcw, Target, Timer, Trophy, XCircle, Zap,
} from 'lucide-react'
import { benchmarkModel, evs, loadFixtures, loadJSON, loadManifest, prepareInputs } from './lib/bench'
import type { Backend, BenchReport, BenchSettings, FixtureSet, Manifest, ModelResult } from './lib/types'
import { EvsBar, LatencyTimeline, SignalPlot, TradeoffScatter, colorFor, type Row } from './components/Charts'

type Mode = 'live' | 'playback'
type AccSource = 'live' | 'heldout'

const REPO_URL = 'https://github.com/Sohamkhrt/openbench'
const fmt = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : '-')

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [fixtures, setFixtures] = useState<FixtureSet | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [mode, setMode] = useState<Mode>('live')
  const [settings, setSettings] = useState<BenchSettings>({ backend: 'wasm', runs: 10, warmup: 5, noise: 0 })
  const [latBudget, setLatBudget] = useState(0.15)
  const [sizeBudget, setSizeBudget] = useState(0.45)
  const [accSource, setAccSource] = useState<AccSource>('live')

  const [results, setResults] = useState<ModelResult[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<{ model: string; pct: number } | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [selected, setSelected] = useState(0)

  useEffect(() => {
    Promise.all([loadManifest(), loadFixtures()])
      .then(([m, f]) => { setManifest(m); setFixtures(f) })
      .catch((e) => setLoadError(String(e)))
  }, [])

  const inputs = useMemo(
    () => (fixtures ? prepareInputs(fixtures.samples, settings.noise) : []),
    [fixtures, settings.noise],
  )

  const rows: Row[] = useMemo(() => {
    if (!manifest) return []
    return results
      .map((r) => {
        const meta = manifest.models.find((m) => m.id === r.modelId)
        if (!meta || r.error) return null
        const accuracy = accSource === 'live' ? r.accuracy : meta.heldOutAccuracy
        return { meta, result: r, accuracy, evs: evs(accuracy, r.p50Ms, meta.sizeMB) }
      })
      .filter((r): r is Row => r !== null)
  }, [results, manifest, accSource])

  const best = rows.length ? rows.reduce((a, b) => (b.evs > a.evs ? b : a)) : null

  async function playback(reason?: string) {
    setMode('playback')
    if (reason) setNotice(reason)
    const report = await loadJSON<BenchReport>('playback.json')
    setSettings((s) => ({ ...report.settings, noise: s.noise }))
    setResults([])
    // Replay one model at a time so the dashboard animates like a live run.
    for (let i = 0; i < report.results.length; i++) {
      const r = report.results[i]
      for (let p = 0; p <= 100; p += 20) {
        setProgress({ model: r.modelId, pct: p })
        await new Promise((res) => setTimeout(res, 70))
      }
      setResults((prev) => [...prev, r])
    }
    setProgress(null)
  }

  async function run() {
    if (!manifest || !fixtures || running) return
    setRunning(true)
    setNotice(null)
    setResults([])
    try {
      if (mode === 'playback') {
        await playback()
        return
      }
      const out: ModelResult[] = []
      const fallbacks: string[] = []
      for (const m of manifest.models) {
        setProgress({ model: m.id, pct: 0 })
        try {
          const r = await benchmarkModel(m, fixtures.samples, inputs, manifest.classes.length, settings, (d, t) =>
            setProgress({ model: m.id, pct: Math.round((d / t) * 100) }),
          )
          if (r.backendUsed !== settings.backend) fallbacks.push(m.name)
          out.push(r)
        } catch (e) {
          throw new Error(`${m.name}: ${e instanceof Error ? e.message : String(e)}`)
        }
        setResults([...out])
      }
      if (fallbacks.length) setNotice(`${settings.backend} has no kernels for ${fallbacks.join(', ')}, so those ran on Wasm.`)
      setProgress(null)
    } catch (e) {
      console.error(e)
      await playback(`Live inference failed (${e instanceof Error ? e.message : e}). Showing recorded playback instead.`)
    } finally {
      setRunning(false)
    }
  }

  function exportReport() {
    const report: BenchReport = { createdAt: new Date().toISOString(), userAgent: navigator.userAgent, settings, results }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `openbench-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  if (loadError) {
    return <div className="p-10 text-rose-400">Could not load models or fixtures: {loadError}</div>
  }

  const sample = fixtures?.samples[selected]

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-400/10 ring-1 ring-emerald-400/30">
              <Activity className="h-4 w-4 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-zinc-100">OpenBench</h1>
              <p className="text-[11px] text-zinc-500">Edge model benchmark, running in your browser</p>
            </div>
          </div>
          <span className="ml-1 hidden rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[11px] text-emerald-300 sm:inline">
            100% client-side · $0 compute
          </span>
          <div className="ml-auto flex items-center gap-2">
            <div className="flex rounded-lg border border-zinc-800 p-0.5 text-xs" role="tablist" aria-label="Mode">
              {(['live', 'playback'] as Mode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  disabled={running}
                  className={`rounded-md px-3 py-1 capitalize transition ${mode === m ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                  {m}
                </button>
              ))}
            </div>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="rounded-lg border border-zinc-800 p-1.5 text-zinc-400 hover:text-zinc-100" aria-label="GitHub repository">
              <Github className="h-4 w-4" />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-4 py-6 sm:px-6 lg:grid-cols-[300px_1fr]">
        {/* Controls */}
        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <section className="card space-y-5 p-4">
            <div>
              <p className="label mb-2">Execution provider</p>
              <div className="grid grid-cols-2 gap-1 rounded-lg border border-zinc-800 p-0.5 text-xs">
                {(['wasm', 'webgpu'] as Backend[]).map((b) => (
                  <button
                    key={b}
                    disabled={running || mode === 'playback'}
                    onClick={() => setSettings({ ...settings, backend: b })}
                    className={`rounded-md py-1.5 font-mono transition ${settings.backend === b ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'} disabled:opacity-50`}
                  >
                    {b}
                  </button>
                ))}
              </div>
            </div>
            <Slider label="Timed runs per sample" value={settings.runs} min={1} max={50} step={1} display={`${settings.runs} × ${fixtures?.samples.length ?? 0}`} onChange={(v) => setSettings({ ...settings, runs: v })} disabled={running || mode === 'playback'} />
            <Slider label="Warm-up inferences" value={settings.warmup} min={0} max={30} step={1} display={`${settings.warmup}`} onChange={(v) => setSettings({ ...settings, warmup: v })} disabled={running || mode === 'playback'} />
            <Slider label="Sensor noise σ" value={settings.noise} min={0} max={0.3} step={0.01} display={`${settings.noise.toFixed(2)} g`} onChange={(v) => setSettings({ ...settings, noise: v })} disabled={running || mode === 'playback'} hint="Adds Gaussian noise to every fixture to stress-test robustness." />

            <button
              onClick={run}
              disabled={running || !manifest}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-500 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:cursor-wait disabled:opacity-60"
            >
              {running ? <RotateCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {running ? 'Benchmarking…' : mode === 'live' ? 'Run benchmark' : 'Play recorded run'}
            </button>
            {progress && (
              <div>
                <div className="mb-1 flex justify-between text-[11px] text-zinc-500">
                  <span>{manifest?.models.find((m) => m.id === progress.model)?.name}</span>
                  <span>{progress.pct}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                  <div className="h-full transition-all" style={{ width: `${progress.pct}%`, background: colorFor(progress.model) }} />
                </div>
              </div>
            )}
          </section>

          <section className="card space-y-5 p-4">
            <p className="label">Deployment budget</p>
            <Slider label="Latency budget (p95)" value={latBudget} min={0.01} max={1} step={0.01} display={`${latBudget.toFixed(2)} ms`} onChange={setLatBudget} />
            <Slider label="Model size budget" value={sizeBudget} min={0.05} max={2} step={0.05} display={`${sizeBudget.toFixed(2)} MB`} onChange={setSizeBudget} />
            <div>
              <p className="label mb-2">Accuracy used in EVS</p>
              <div className="grid grid-cols-2 gap-1 rounded-lg border border-zinc-800 p-0.5 text-xs">
                {([['live', 'Fixtures (live)'], ['heldout', 'Held-out set']] as [AccSource, string][]).map(([k, l]) => (
                  <button key={k} onClick={() => setAccSource(k)} className={`rounded-md py-1.5 transition ${accSource === k ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="card p-4 text-xs leading-relaxed text-zinc-400">
            <p className="label mb-2">Edge Viability Score</p>
            <p className="font-mono text-[13px] text-zinc-200">EVS = (Acc × 100) / (Latency<sub>ms</sub> × Size<sub>MB</sub>)</p>
            <p className="mt-2">Latency is the median per-call time of <span className="font-mono">session.run()</span>, measured with <span className="font-mono">performance.now()</span>. Higher is better.</p>
          </section>
        </aside>

        {/* Results */}
        <div className="min-w-0 space-y-5">
          {notice && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {notice}
            </div>
          )}
          {mode === 'playback' && (
            <div className="flex items-start gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-200">
              <Info className="mt-0.5 h-4 w-4 shrink-0" /> Playback mode replays a benchmark recorded in headless Chromium (Wasm, 1 thread). Switch to Live to measure your own device.
            </div>
          )}

          {/* Model cards */}
          <div className="grid gap-4 md:grid-cols-3">
            {manifest?.models.map((m) => {
              const row = rows.find((r) => r.meta.id === m.id)
              const passLat = row ? row.result.p95Ms <= latBudget : null
              const passSize = m.sizeMB <= sizeBudget
              const isBest = best && row && best.meta.id === m.id
              return (
                <div key={m.id} className={`card relative p-4 ${isBest ? 'ring-1 ring-emerald-400/50' : ''}`}>
                  <div className="absolute inset-x-0 top-0 h-0.5 rounded-t-xl" style={{ background: colorFor(m.id) }} />
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-semibold text-zinc-100">{m.name}</h3>
                      <p className="mt-0.5 text-[11px] text-zinc-500">{m.description}</p>
                    </div>
                    <span className="rounded border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-400">{m.precision}</span>
                  </div>
                  <div className="mt-4 flex items-end justify-between">
                    <div>
                      <p className="label">EVS</p>
                      <p className="font-mono text-3xl font-semibold tabular-nums text-zinc-50">{row ? Math.round(row.evs).toLocaleString() : '-'}</p>
                    </div>
                    {isBest && (
                      <span className="flex items-center gap-1 rounded-full bg-emerald-400/10 px-2 py-0.5 text-[11px] text-emerald-300">
                        <Trophy className="h-3 w-3" /> Most viable
                      </span>
                    )}
                  </div>
                  <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                    <Stat icon={<Target className="h-3.5 w-3.5" />} label="Accuracy" value={row ? `${fmt(row.accuracy * 100, 1)}%` : '-'} />
                    <Stat icon={<Timer className="h-3.5 w-3.5" />} label="p50 latency" value={row ? `${fmt(row.result.p50Ms, 3)} ms` : '-'} />
                    <Stat icon={<HardDrive className="h-3.5 w-3.5" />} label="Size" value={`${fmt(m.sizeMB, 3)} MB`} />
                    <Stat icon={<Zap className="h-3.5 w-3.5" />} label="p95 latency" value={row ? `${fmt(row.result.p95Ms, 3)} ms` : '-'} />
                  </dl>
                  <div className="mt-4 flex gap-2 text-[11px]">
                    <Budget ok={passLat} label="Latency" />
                    <Budget ok={passSize} label="Size" />
                  </div>
                </div>
              )
            })}
          </div>

          {rows.length === 0 && !running && (
            <div className="card grid place-items-center gap-2 p-10 text-center">
              <Gauge className="h-8 w-8 text-zinc-600" />
              <p className="text-sm text-zinc-300">Press <b>Run benchmark</b> to load three ONNX models and profile them on {fixtures?.samples.length ?? 20} road-vibration samples.</p>
              <p className="text-xs text-zinc-500">Everything runs locally through onnxruntime-web (WebAssembly). No data leaves your device.</p>
            </div>
          )}

          {rows.length > 0 && (
            <div className="grid gap-4 xl:grid-cols-2">
              <Panel title="Edge Viability Score" icon={<Trophy className="h-4 w-4" />}><EvsBar rows={rows} /></Panel>
              <Panel title="Accuracy vs latency (bubble = size)" icon={<Target className="h-4 w-4" />}><TradeoffScatter rows={rows} /></Panel>
              <div className="xl:col-span-2">
                <Panel title="Per-inference latency (spikes show jitter)" icon={<Timer className="h-4 w-4" />}><LatencyTimeline rows={rows} budget={latBudget} /></Panel>
              </div>
            </div>
          )}

          {/* Signal inspector */}
          {fixtures && sample && (
            <Panel title="Signal inspector" icon={<Activity className="h-4 w-4" />}
              right={
                <select value={selected} onChange={(e) => setSelected(+e.target.value)} className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-xs text-zinc-200">
                  {fixtures.samples.map((s, i) => <option key={s.id} value={i}>{s.id}</option>)}
                </select>
              }>
              <SignalPlot data={Array.from(inputs[selected] ?? sample.data)} length={fixtures.length} />
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-zinc-500">Ground truth:</span>
                <span className="rounded bg-zinc-800 px-2 py-0.5 font-mono text-zinc-100">{sample.label}</span>
                {rows.map((r) => {
                  const p = r.result.predictions[selected]
                  const ok = p === sample.labelIndex
                  return (
                    <span key={r.meta.id} className={`flex items-center gap-1 rounded border px-2 py-0.5 font-mono ${ok ? 'border-emerald-500/30 text-emerald-300' : 'border-rose-500/30 text-rose-300'}`}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: colorFor(r.meta.id) }} />
                      {r.meta.name}: {manifest?.classes[p] ?? '?'} ({fmt((r.result.confidences[selected] ?? 0) * 100, 0)}%)
                    </span>
                  )
                })}
              </div>
            </Panel>
          )}

          {/* Detail table */}
          {rows.length > 0 && (
            <Panel title="Full results" icon={<Cpu className="h-4 w-4" />}
              right={<button onClick={exportReport} className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:text-zinc-50"><Download className="h-3.5 w-3.5" /> Export JSON</button>}>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-xs">
                  <thead className="text-zinc-500">
                    <tr className="border-b border-zinc-800">
                      {['Model', 'Backend', 'Fixture acc', 'Held-out acc', 'Calls/timing', 'Mean', 'p50', 'p95', 'Max', 'σ', 'Load', 'Size', 'EVS'].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums text-zinc-300">
                    {rows.map((r) => (
                      <tr key={r.meta.id} className="border-b border-zinc-800/60">
                        <td className="py-2 pr-3 font-sans text-zinc-100">{r.meta.name}</td>
                        <td className="pr-3">{r.result.backendUsed}</td>
                        <td className="pr-3">{r.result.correct}/{r.result.total}</td>
                        <td className="pr-3">{fmt(r.meta.heldOutAccuracy * 100, 1)}%</td>
                        <td className="pr-3">×{r.result.callsPerTiming ?? 1}</td>
                        <td className="pr-3">{fmt(r.result.meanMs, 3)}</td>
                        <td className="pr-3">{fmt(r.result.p50Ms, 3)}</td>
                        <td className="pr-3">{fmt(r.result.p95Ms, 3)}</td>
                        <td className="pr-3">{fmt(r.result.maxMs, 3)}</td>
                        <td className="pr-3">{fmt(r.result.stdMs, 3)}</td>
                        <td className="pr-3">{fmt(r.result.loadMs, 1)}</td>
                        <td className="pr-3">{fmt(r.meta.sizeMB, 3)}</td>
                        <td className="pr-3 text-zinc-50">{Math.round(r.evs).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-[11px] text-zinc-500">Latencies in ms per call. Each timing averages a calibrated batch of back-to-back calls because browsers round performance.now() to about 0.1 ms. Held-out accuracy is from {manifest?.models[0]?.heldOutSamples ?? 1000} samples at training time; fixture accuracy is measured live on this page.</p>
              </div>
            </Panel>
          )}
        </div>
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 text-[11px] text-zinc-600 sm:px-6">
        OpenBench · MIT licensed · onnxruntime-web, React, Recharts · <a className="underline hover:text-zinc-400" href={REPO_URL}>source</a>
      </footer>
    </div>
  )
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; display: string; onChange: (v: number) => void; disabled?: boolean; hint?: string }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex justify-between text-xs">
        <span className="text-zinc-400">{props.label}</span>
        <span className="font-mono text-zinc-200">{props.display}</span>
      </div>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} disabled={props.disabled} onChange={(e) => props.onChange(+e.target.value)} className="disabled:opacity-40" />
      {props.hint && <p className="mt-1 text-[11px] text-zinc-600">{props.hint}</p>}
    </label>
  )
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div>
      <dt className="flex items-center gap-1 text-zinc-500">{icon}{label}</dt>
      <dd className="font-mono tabular-nums text-zinc-200">{value}</dd>
    </div>
  )
}

function Budget({ ok, label }: { ok: boolean | null; label: string }) {
  if (ok === null) return <span className="rounded border border-zinc-800 px-1.5 py-0.5 text-zinc-600">{label} budget</span>
  return ok ? (
    <span className="flex items-center gap-1 rounded border border-emerald-500/30 px-1.5 py-0.5 text-emerald-300"><CheckCircle2 className="h-3 w-3" />{label} OK</span>
  ) : (
    <span className="flex items-center gap-1 rounded border border-rose-500/30 px-1.5 py-0.5 text-rose-300"><XCircle className="h-3 w-3" />{label} over</span>
  )
}

function Panel({ title, icon, right, children }: { title: string; icon: React.ReactNode; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-sm font-medium text-zinc-200"><span className="text-zinc-500">{icon}</span>{title}</h2>
        {right}
      </div>
      {children}
    </section>
  )
}
