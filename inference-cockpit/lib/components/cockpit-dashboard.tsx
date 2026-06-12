"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "@/lib/api"
import { InspectionResult, RunningStats } from "@/types/inspection"
import { ControlBar }         from "@/lib/components/control-bar"
import { ConveyorPanel }      from "@/lib/components/conveyor-panel"
import { RecommendationCard } from "@/lib/components/recommendation-card"
import { StatsPanel }         from "@/lib/components/stats-panel"
import { ConfidenceChart }    from "@/lib/components/confidence-chart"
import { InspectionLog }      from "@/lib/components/inspection-log"
import { AlertBanner }        from "@/lib/components/alert-banner"

export function CockpitDashboard() {
  const [isRunning,  setIsRunning]  = useState(false)
  const [speed,      setSpeed]      = useState(2000)
  const [current,    setCurrent]    = useState<InspectionResult | null>(null)
  const [history,    setHistory]    = useState<InspectionResult[]>([])
  const [stats,      setStats]      = useState<RunningStats | null>(null)
  const [isAlert,    setIsAlert]    = useState(false)
  const [connected,  setConnected]  = useState(false)
  const intervalRef   = useRef<ReturnType<typeof setInterval> | null>(null)
  const alertTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const check = async () => {
      try { await api.health(); setConnected(true) }
      catch { setConnected(false) }
    }
    check()
    const t = setInterval(check, 5000)
    return () => clearInterval(t)
  }, [])

  const triggerAlert = useCallback(() => {
    setIsAlert(true)
    if (alertTimerRef.current) clearTimeout(alertTimerRef.current)
    alertTimerRef.current = setTimeout(() => setIsAlert(false), 3000)
  }, [])

  useEffect(() => {
    if (!isRunning) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      return
    }
    const tick = async () => {
      try {
        const raw    = await api.predictNext()
        const result: InspectionResult = { ...raw, timestamp: Date.now() }
        setCurrent(result)
        setHistory(prev => [result, ...prev].slice(0, 50))
        if (!result.correct) triggerAlert()
        const s = await api.getStats()
        setStats(s)
      } catch {
        setConnected(false)
        setIsRunning(false)
      }
    }
    tick()
    intervalRef.current = setInterval(tick, speed)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [isRunning, speed, triggerAlert])

  const handleReset = async () => {
    setIsRunning(false)
    setCurrent(null)
    setHistory([])
    setStats(null)
    setIsAlert(false)
    try { await api.reset() } catch { /* offline */ }
  }

  return (
    <div className="min-h-screen bg-zinc-950 p-4 flex flex-col gap-3">
      <AlertBanner result={current} visible={isAlert} />

      {/* ── Header ── */}
      <div className="flex items-center gap-3 px-1">
        <div className="w-8 h-8 bg-zinc-800 border border-zinc-700 rounded-lg flex items-center justify-center text-zinc-300 text-sm font-bold font-mono">Q</div>
        <div>
          <h1 className="text-sm font-semibold text-zinc-100 leading-none tracking-wide uppercase">CNN Defect Inspector</h1>
          <p className="text-xs text-zinc-500 mt-0.5">4-Klassen Defektklassifikation · ResNet-18 · Eigene Produktionsdaten</p>
        </div>
        <div className="ml-auto text-xs text-zinc-600 font-mono">
          {new Date().toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit", year: "numeric" })}
        </div>
      </div>

      {/* ── Control bar ── */}
      <ControlBar
        isRunning={isRunning}
        speed={speed}
        connected={connected}
        onToggle={() => setIsRunning(r => !r)}
        onReset={handleReset}
        onSpeed={setSpeed}
      />

      {/* ── Main two-column layout ── */}
      <div className="grid grid-cols-[360px_1fr] gap-0">

        {/* LEFT — Inferenz */}
        <div className="flex flex-col gap-1 pr-4 border-r border-zinc-800">
          <p className="text-[9px] text-zinc-600 font-mono uppercase tracking-widest mb-1">Inferenz</p>
          <ConveyorPanel result={current} stats={stats} />
        </div>

        {/* RIGHT — Prozesssteuerung */}
        <div className="flex flex-col gap-3 pl-4">
          <p className="text-[9px] text-zinc-600 font-mono uppercase tracking-widest mb-0">Prozesssteuerung</p>
          <StatsPanel stats={stats} />
          <RecommendationCard result={current} />
          <ConfidenceChart history={history} />
        </div>
      </div>

      {/* ── Full-width log ── */}
      <InspectionLog history={history} />
    </div>
  )
}
