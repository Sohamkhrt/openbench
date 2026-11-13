import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, Download, Eye, FlaskConical, Gauge, Github, HardDrive, Info, Layers, ListChecks, Play,
  RotateCcw, ShieldCheck, Upload, X, Sparkles, Table2, Target, Timer, Trophy,
} from 'lucide-react'
import { benchmarkModel, probeModel, warmRuntime, loadFixtures, loadJSON, loadManifest, prepareInputs, type Phase } from './lib/bench'
import { METRICS, SECTIONS, type SectionId, colorFor, fillFor, makeRow, metricById, type Row } from './lib/metrics'
import type { Backend, BenchReport, BenchSettings, FixtureSet, Manifest, ModelMeta, ModelResult } from './lib/types'
import { ConfusionGrid } from './components/Confusion'
import { Lab } from './components/Lab'
import { LatencyTimeline, TradeoffScatter } from './components/Charts'
import { GlobalHatchDefs, MetricCard } from './components/Leaderboard'
import { Theater, type WatchState } from './components/Theater'
import { Recommendations, type Constraints } from './components/Verdicts'

type Mode = 'live' | 'playback'
type AccSource = 'live' | 'heldout'

const REPO_URL = 'https://github.com/Sohamkhrt/openbench'
const fmt = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : '-')
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

const SPEEDS = [
  { label: 'Slow', ms: 220 },
  { label: 'Normal', ms: 70 },
  { label: 'Skip', ms: 0 },
]

const SECTION_ICON: Record<string, React.ReactNode> = {
  live: <Eye className="h-3.5 w-3.5" />,
  lab: <FlaskConical className="h-3.5 w-3.5" />,
  overall: <Trophy className="h-3.5 w-3.5" />,
  quality: <Target className="h-3.5 w-3.5" />,
  speed: <Timer className="h-3.5 w-3.5" />,
  footprint: <HardDrive className="h-3.5 w-3.5" />,
  robustness: <ShieldCheck className="h-3.5 w-3.5" />,
  tradeoffs: <Layers className="h-3.5 w-3.5" />,
  pick: <Sparkles className="h-3.5 w-3.5" />,
  raw: <Table2 className="h-3.5 w-3.5" />,
}

const NAV = [
  { id: 'live', title: 'Live inference' },
  { id: 'lab', title: 'Try it yourself' },
  ...SECTIONS.map((s) => ({ id: s.id, title: s.title })),
  { id: 'tradeoffs', title: 'Trade-offs' },
  { id: 'pick', title: 'What should I pick?' },
  { id: 'raw', title: 'Raw data' },
]

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [fixtures, setFixtures] = useState<FixtureSet | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [mode, setMode] = useState<Mode>('live')
  const [settings, setSettings] = useState<BenchSettings>({ backend: 'wasm', runs: 10, warmup: 5, noise: 0, watchDelay: 70 })
  const [accSource, setAccSource] = useState<AccSource>('heldout')
  const [constraints, setConstraints] = useState<Constraints>({ deviceId: 'pico', minAccuracy: 0.9, latencyBudgetMs: 0.5 })
  const [enabled, setEnabled] = useState<Record<string, boolean>>({})
  const [custom, setCustom] = useState<ModelMeta[]>([])
  const customCount = useRef(0)

  const [results, setResults] = useState<ModelResult[]>([])
  const [recorded, setRecorded] = useState(false)
  const [running, setRunning] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [selected, setSelected] = useState(10)
  const [highlight, setHighlight] = useState<string | null>(null)

  // live inference state
  const [watch, setWatch] = useState<WatchState | null>(null)
  const [phases, setPhases] = useState<Record<string, Phase>>({})
  const [cells, setCells] = useState<Record<string, (boolean | null)[]>>({})
  const [progress, setProgress] = useState<Record<string, number>>({})
  const tickRef = useRef(0)

  useEffect(() => {
    Promise.all([loadManifest(), loadFixtures(), loadJSON<BenchReport>('playback.json').catch(() => null)])
      .then(([m, f, rep]) => {
        setManifest(m)
        setFixtures(f)
        setEnabled(Object.fromEntries(m.models.map((x) => [x.id, true])))
        if (rep) {
          // Show a recorded run immediately so the page is never empty.
          setResults(rep.results)
          setRecorded(true)
          setCells(fillCells(rep.results, f))
        }
      })
      .catch((e) => setLoadError(String(e)))
  }, [])

  const inputs = useMemo(() => (fixtures ? prepareInputs(fixtures.samples, settings.noise) : []), [fixtures, settings.noise])

  const allModels = useMemo(() => [...(manifest?.models ?? []), ...custom], [manifest, custom])
  const activeModels = useMemo(() => allModels.filter((m) => enabled[m.id]), [allModels, enabled])

  const rows: Row[] = useMemo(() => {
    if (!manifest) return []
    return results
      .map((r) => {
        const meta = allModels.find((m) => m.id === r.modelId)
        return meta && enabled[meta.id] && !r.error ? makeRow(meta, r, accSource) : null
      })
      .filter((r): r is Row => r !== null)
  }, [results, manifest, allModels, accSource, enabled])

  function fillCells(rs: ModelResult[], f: FixtureSet) {
    return Object.fromEntries(rs.map((r) => [r.modelId, r.predictions.map((p, i) => p === f.samples[i].labelIndex)]))
  }

  function resetLive() {
    setResults([])
    setRecorded(false)
    setWatch(null)
    setPhases({})
    setProgress({})
    setCells({})
  }

  const step = (modelId: string, sampleIndex: number, probs: number[], ms: number, label: number) => {
    tickRef.current++
    setWatch({ modelId, sampleIndex, probs, ms, tick: tickRef.current })
    const k = probs.indexOf(Math.max(...probs))
    setCells((prev) => {
      const row = [...(prev[modelId] ?? Array(20).fill(null))]
      row[sampleIndex] = k === label
      return { ...prev, [modelId]: row }
    })
  }

  async function playback(reason?: string) {
    if (!fixtures) return
    setMode('playback')
    if (reason) setNotice(reason)
    const report = await loadJSON<BenchReport>('playback.json')
    resetLive()
    const delay = Math.min(settings.watchDelay, 40)
    for (const r of report.results.filter((x) => enabled[x.modelId])) {
      setPhases((p) => ({ ...p, [r.modelId]: 'watching' }))
      if (delay > 0) {
        for (let i = 0; i < fixtures.samples.length; i++) {
          step(r.modelId, i, r.probs?.[i] ?? [], 0, fixtures.samples[i].labelIndex)
          await sleep(delay)
        }
      }
      setPhases((p) => ({ ...p, [r.modelId]: 'done' }))
      setResults((prev) => [...prev, r])
    }
    setCells(fillCells(report.results, fixtures))
    setRecorded(true)
  }

  async function run() {
    if (!manifest || !fixtures || running) return
    setRunning(true)
    setNotice(null)
    document.getElementById('live')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    try {
      if (mode === 'playback') {
        await playback()
        return
      }
      resetLive()
      const out: ModelResult[] = []
      const fallbacks: string[] = []
      // Smallest model warms up the Wasm runtime so cold start compares models, not the runtime.
      await warmRuntime([...manifest.models].sort((x, y) => x.sizeMB - y.sizeMB)[0])
      for (const m of activeModels) {
        try {
          const r = await benchmarkModel(m, fixtures.samples, inputs, manifest.classes.length, settings, {
            onPhase: (id, ph) => setPhases((p) => ({ ...p, [id]: ph })),
            onVisual: (s) => step(s.modelId, s.sampleIndex, s.probs, s.ms, fixtures.samples[s.sampleIndex].labelIndex),
            onProgress: (id, pct) => setProgress((p) => ({ ...p, [id]: pct })),
          })
          if (r.backendUsed !== settings.backend) fallbacks.push(m.name)
          out.push(r)
        } catch (e) {
          throw new Error(`${m.name}: ${e instanceof Error ? e.message : String(e)}`)
        }
        setResults([...out])
      }
      setCells(fillCells(out, fixtures))
      if (fallbacks.length) setNotice(`${settings.backend} has no kernels for ${fallbacks.join(', ')}, so those ran on Wasm.`)
    } catch (e) {
      console.error(e)
      await playback(`Live inference failed (${e instanceof Error ? e.message : e}). Showing recorded playback instead.`)
    } finally {
      setRunning(false)
    }
  }

  async function addModel(file: File) {
    setNotice(null)
    const id = `custom-${++customCount.current}`
    const meta: ModelMeta = {
      id, name: file.name.replace(/\.onnx$/i, '').slice(0, 24) || 'My model', family: 'custom', custom: true,
      description: `Uploaded ${file.name}; quality is measured on the 20 live fixtures`,
      precision: 'fp32', file: URL.createObjectURL(file), sizeMB: file.size / (1024 * 1024),
      params: Math.max(1, Math.round(file.size / 4)),
      heldOutAccuracy: 0, heldOutMacroF1: 0, heldOutHazardRecall: 0, heldOutRecall: [], heldOutSamples: 0,
    }
    try {
      const info = await probeModel(meta)
      setCustom((c) => [...c, meta])
      setEnabled((e) => ({ ...e, [id]: true }))
      setMode('live')
      setNotice(`Added "${meta.name}" (input [${info.shape.join(', ')}], output "${info.output}"${info.needsSoftmax ? ', softmax applied' : ''}). Press Run to benchmark it against the built-in models.`)
    } catch (e) {
      URL.revokeObjectURL(meta.file)
      setNotice(`Could not use ${file.name}. ${e instanceof Error ? e.message : e} OpenBench needs a float32 input of 384 values ([1,384] or [1,3,128]) and an output with 4 class scores.`)
    }
  }

  function removeModel(id: string) {
    setCustom((c) => c.filter((m) => m.id !== id))
    setResults((r) => r.filter((x) => x.modelId !== id))
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

  if (loadError) return <div className="p-10 text-rose-400">Could not load models or fixtures: {loadError}</div>
  if (!manifest || !fixtures) return <div className="grid min-h-screen place-items-center text-sm text-zinc-500">Loading models…</div>

  const evsMetric = metricById('evs')
  const disabled = running || mode === 'playback'

  return (
    <div className="min-h-screen">
      <GlobalHatchDefs ids={allModels.map((m) => m.id)} />
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3 px-4 py-3 sm:px-6">
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
                <button key={m} onClick={() => setMode(m)} disabled={running}
                  className={`rounded-md px-3 py-1 capitalize transition ${mode === m ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
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

      <div className="mx-auto max-w-[1400px] px-4 sm:px-6">
        {/* Hero + controls */}
        <section className="grid gap-5 py-6 lg:grid-cols-[1fr_380px]">
          <div className="flex flex-col justify-center">
            <h2 className="text-2xl font-semibold tracking-tight text-zinc-50 sm:text-3xl">Which model should go on your edge device?</h2>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-400">
              A car-mounted accelerometer has to tell <b className="text-zinc-200">smooth road</b>, <b className="text-zinc-200">rough road</b>, <b className="text-zinc-200">potholes</b> and <b className="text-zinc-200">speed bumps</b> apart, on a chip with little memory and no internet.
              OpenBench loads {manifest.models.length} candidate models into your browser, runs every one on the same 20 sensor recordings, and measures how accurate, fast, small and noise-tolerant each is. Nothing leaves your device.
            </p>
            <ol className="mt-4 grid gap-2 text-xs sm:grid-cols-3">
              {[
                ['1', 'Watch', 'See each model read the signal and make its call, live.'],
                ['2', 'Compare', 'Ranked charts for 14 metrics, each explained in plain English. Add your own model too.'],
                ['3', 'Pick', 'Set your device and budget; get a recommendation and why.'],
              ].map(([n, t, d]) => (
                <li key={n} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                  <span className="font-mono text-emerald-400">{n}</span> <b className="text-zinc-200">{t}</b>
                  <p className="mt-1 text-zinc-500">{d}</p>
                </li>
              ))}
            </ol>
          </div>

          <div className="card space-y-4 p-4">
            <button onClick={run} disabled={running}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-500 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:cursor-wait disabled:opacity-60">
              {running ? <RotateCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {running ? 'Benchmarking…' : mode === 'live' ? `Run benchmark on ${activeModels.length} models` : 'Replay recorded run'}
            </button>
            <div>
              <p className="label mb-1.5">Animation speed</p>
              <Segmented options={SPEEDS.map((s) => ({ key: String(s.ms), label: s.label }))} value={String(settings.watchDelay)} onChange={(v) => setSettings({ ...settings, watchDelay: +v })} disabled={running} />
            </div>
            <Slider label="Timed runs per sample" value={settings.runs} min={1} max={50} step={1} display={`${settings.runs} × ${fixtures.samples.length}`} onChange={(v) => setSettings({ ...settings, runs: v })} disabled={disabled} />
            <Slider label="Warm-up inferences" value={settings.warmup} min={0} max={30} step={1} display={`${settings.warmup}`} onChange={(v) => setSettings({ ...settings, warmup: v })} disabled={disabled} />
            <Slider label="Extra sensor noise σ" value={settings.noise} min={0} max={0.3} step={0.01} display={`${settings.noise.toFixed(2)} g`} onChange={(v) => setSettings({ ...settings, noise: v })} disabled={disabled} />
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="label mb-1.5">Provider</p>
                <Segmented options={[{ key: 'wasm', label: 'wasm' }, { key: 'webgpu', label: 'webgpu' }]} value={settings.backend} onChange={(v) => setSettings({ ...settings, backend: v as Backend })} disabled={disabled} mono />
              </div>
              <div>
                <p className="label mb-1.5">EVS accuracy</p>
                <Segmented options={[{ key: 'heldout', label: 'Held-out' }, { key: 'live', label: 'Fixtures' }]} value={accSource} onChange={(v) => setAccSource(v as AccSource)} />
              </div>
            </div>
          </div>
        </section>

        {notice && (
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {notice}
          </div>
        )}
        {recorded && !running && (
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-200">
            <Info className="mt-0.5 h-4 w-4 shrink-0" /> Showing a run recorded in headless Chromium (Wasm, 1 thread). {mode === 'live' ? 'Press Run to measure your own device.' : 'Switch to Live to measure your own device.'}
          </div>
        )}

        {/* model picker */}
        <div className="mb-5 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-zinc-500">Comparing <b className="text-zinc-200">{activeModels.length}</b> of {allModels.length} models</span>
          {allModels.map((m) => (
            <button key={m.id} onClick={() => setEnabled({ ...enabled, [m.id]: !enabled[m.id] })} disabled={running}
              onMouseEnter={() => setHighlight(m.id)} onMouseLeave={() => setHighlight(null)}
              className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition ${enabled[m.id] ? 'border-zinc-700 bg-zinc-900 text-zinc-200' : 'border-zinc-800 text-zinc-600 line-through'}`}>
              <svg width="10" height="10" aria-hidden><rect width="10" height="10" rx="2" fill={fillFor(m)} opacity={enabled[m.id] ? 1 : 0.3} /></svg>
              {m.name}
              {m.custom && !running && (
                <span role="button" tabIndex={0} aria-label={`Remove ${m.name}`} onClick={(e) => { e.stopPropagation(); removeModel(m.id) }} onKeyDown={(e) => { if (e.key === 'Enter') removeModel(m.id) }} className="ml-0.5 rounded-full text-zinc-500 hover:text-rose-300">
                  <X className="h-3 w-3" />
                </span>
              )}
            </button>
          ))}
          <label className={`flex cursor-pointer items-center gap-1.5 rounded-full border border-dashed border-emerald-500/40 px-2.5 py-1 text-xs text-emerald-300 transition hover:bg-emerald-500/10 ${running ? 'pointer-events-none opacity-50' : ''}`}>
            <Upload className="h-3 w-3" /> Add your .onnx
            <input type="file" accept=".onnx" className="sr-only" disabled={running} onChange={(e) => { const f = e.target.files?.[0]; if (f) void addModel(f); e.target.value = '' }} />
          </label>
          <span className="ml-auto text-[11px] text-zinc-600">Hatched bars = INT8 quantized</span>
        </div>

        <div className="grid gap-6 lg:grid-cols-[200px_1fr]">
          {/* section nav */}
          <nav className="hidden lg:block">
            <ul className="sticky top-20 space-y-0.5 text-[13px]">
              {NAV.map((n) => (
                <li key={n.id}>
                  <a href={`#${n.id}`} className="flex items-center gap-2 rounded-md px-2 py-1.5 text-zinc-400 transition hover:bg-zinc-900 hover:text-zinc-100">
                    <span className="text-zinc-600">{SECTION_ICON[n.id]}</span>{n.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          <main className="min-w-0 space-y-10 pb-16">
            <Theater
              models={activeModels} classes={manifest.classes} samples={fixtures.samples} inputs={inputs} length={fixtures.length}
              watch={watch} phases={phases} cells={cells} progress={progress} running={running} sweepMs={settings.watchDelay}
              selected={selected} onPickSample={(i) => { setSelected(i); if (!running) setWatch(null) }}
            />

            <Section id="lab" title="Try it yourself" blurb="Shape a road signal with the sliders, or record one from your phone, and watch every model react.">
              <Lab models={activeModels} classes={manifest.classes} disabled={running} />
            </Section>

            {rows.length === 0 ? (
              <div className="card grid place-items-center gap-2 p-10 text-center">
                <Gauge className="h-8 w-8 text-zinc-600" />
                <p className="text-sm text-zinc-300">{running ? 'Results appear here as each model finishes.' : 'Press Run to fill the leaderboard.'}</p>
              </div>
            ) : (
              <>
                {SECTIONS.map((s) => (
                  <Section key={s.id} id={s.id} title={s.title} blurb={s.blurb}>
                    {s.id === 'overall' ? (
                      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
                        <MetricCard metric={evsMetric} rows={rows} highlight={highlight} />
                        <div className="card p-4 text-sm leading-relaxed text-zinc-400">
                          <h3 className="text-[15px] font-semibold text-zinc-100">How to read the EVS</h3>
                          <p className="mt-2 font-mono text-[13px] text-zinc-200">EVS = (Accuracy × 100) / (Latency<sub>ms</sub> × Size<sub>MB</sub>)</p>
                          <p className="mt-3">Each factor counts equally, so halving the latency or halving the size doubles the score. That makes EVS great at spotting efficient models, and harsh on big ones.</p>
                          <p className="mt-3">Its blind spot: a tiny model that is often wrong can still score high. That is why OpenBench also ranks hazard recall and lets you set a minimum accuracy under <a href="#pick" className="text-emerald-400 hover:underline">What should I pick?</a></p>
                          <ul className="mt-3 space-y-1 text-xs">
                            <li><b className="text-zinc-200">Latency</b> is the median per-call time measured live in this browser.</li>
                            <li><b className="text-zinc-200">Accuracy</b> is {accSource === 'heldout' ? 'from 1,000 held-out windows' : 'from the 20 live fixtures'} (switch it in the controls).</li>
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                          {METRICS.filter((m) => m.section === (s.id as SectionId)).map((m) => <MetricCard key={m.id} metric={m} rows={rows} highlight={highlight} />)}
                        </div>
                        {s.id === 'quality' && <ConfusionGrid rows={rows} classes={manifest.classes} />}
                      </>
                    )}
                  </Section>
                ))}

                <Section id="tradeoffs" title="Trade-offs" blurb="Up and to the left is better. Bubble size is file size; the x-axis is logarithmic.">
                  <div className="card p-4"><TradeoffScatter rows={rows} /></div>
                </Section>

                <Section id="pick" title="What should I pick?" blurb="Tell OpenBench about your hardware and it will explain what each choice means for you.">
                  <Recommendations rows={rows} c={constraints} setC={setConstraints} />
                </Section>

                <Section id="raw" title="Raw data" blurb="Every timed inference, and the full table behind the charts."
                  right={<button onClick={exportReport} className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:text-zinc-50"><Download className="h-3.5 w-3.5" /> Export JSON</button>}>
                  <div className="card p-4">
                    <h3 className="mb-2 flex items-center gap-2 text-sm text-zinc-200"><ListChecks className="h-4 w-4 text-zinc-500" />Per-inference latency (log scale; spikes are jitter)</h3>
                    <LatencyTimeline rows={rows} budget={constraints.latencyBudgetMs} />
                  </div>
                  <div className="card mt-4 overflow-x-auto p-4">
                    <table className="w-full min-w-[900px] text-left text-xs">
                      <thead className="text-zinc-500">
                        <tr className="border-b border-zinc-800">
                          {['Model', 'Backend', 'Fixtures', 'Held-out acc', 'Hazard rec.', 'F1', 'Noisy acc', 'Calls/timing', 'Mean', 'p50', 'p95', 'Max', 'σ', 'Load', 'Size MB', 'Params', 'EVS'].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}
                        </tr>
                      </thead>
                      <tbody className="font-mono tabular-nums text-zinc-300">
                        {rows.map((r) => (
                          <tr key={r.meta.id} className="border-b border-zinc-800/60">
                            <td className="whitespace-nowrap py-2 pr-3 font-sans text-zinc-100"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: colorFor(r.meta.id) }} />{r.meta.name}</td>
                            <td className="pr-3">{r.result.backendUsed}</td>
                            <td className="pr-3">{r.result.correct}/{r.result.total}</td>
                            <td className="pr-3">{fmt(r.meta.heldOutAccuracy * 100, 1)}%</td>
                            <td className="pr-3">{fmt(r.meta.heldOutHazardRecall * 100, 1)}%</td>
                            <td className="pr-3">{fmt(r.meta.heldOutMacroF1, 3)}</td>
                            <td className="pr-3">{fmt(r.result.noisyAccuracy * 100, 0)}%</td>
                            <td className="pr-3">×{r.result.callsPerTiming ?? 1}</td>
                            <td className="pr-3">{fmt(r.result.meanMs, 4)}</td>
                            <td className="pr-3">{fmt(r.result.p50Ms, 4)}</td>
                            <td className="pr-3">{fmt(r.result.p95Ms, 4)}</td>
                            <td className="pr-3">{fmt(r.result.maxMs, 3)}</td>
                            <td className="pr-3">{fmt(r.result.stdMs, 4)}</td>
                            <td className="pr-3">{fmt(r.result.loadMs, 0)}</td>
                            <td className="pr-3">{fmt(r.meta.sizeMB, 3)}</td>
                            <td className="pr-3">{r.meta.params.toLocaleString()}</td>
                            <td className="pr-3 text-zinc-50">{Math.round(r.evs).toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-2 text-[11px] text-zinc-500">Latencies in ms per call. Each timing averages a calibrated batch of back-to-back calls because browsers round performance.now() to about 0.1 ms.</p>
                  </div>
                </Section>
              </>
            )}
          </main>
        </div>
      </div>

      <footer className="mx-auto max-w-[1400px] border-t border-zinc-900 px-4 py-6 text-[11px] text-zinc-600 sm:px-6">
        OpenBench · MIT licensed · onnxruntime-web, React, Recharts, scikit-learn · <a className="underline hover:text-zinc-400" href={REPO_URL}>source</a>
      </footer>
    </div>
  )
}

function Section({ id, title, blurb, right, children }: { id: string; title: string; blurb: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20">
      <div className="mb-3 flex items-end justify-between gap-3 border-b border-zinc-800/80 pb-2">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight text-zinc-100"><span className="text-zinc-500">{SECTION_ICON[id]}</span>{title}</h2>
          <p className="mt-0.5 text-xs text-zinc-500">{blurb}</p>
        </div>
        {right}
      </div>
      {children}
    </section>
  )
}

function Segmented({ options, value, onChange, disabled, mono }: { options: { key: string; label: string }[]; value: string; onChange: (k: string) => void; disabled?: boolean; mono?: boolean }) {
  return (
    <div className="flex rounded-lg border border-zinc-800 p-0.5 text-xs">
      {options.map((o) => (
        <button key={o.key} disabled={disabled} onClick={() => onChange(o.key)}
          className={`flex-1 rounded-md py-1.5 transition ${mono ? 'font-mono' : ''} ${value === o.key ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'} disabled:opacity-50`}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; display: string; onChange: (v: number) => void; disabled?: boolean }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex justify-between text-xs">
        <span className="text-zinc-400">{props.label}</span>
        <span className="font-mono text-zinc-200">{props.display}</span>
      </div>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} disabled={props.disabled} onChange={(e) => props.onChange(+e.target.value)} className="disabled:opacity-40" />
    </label>
  )
}
