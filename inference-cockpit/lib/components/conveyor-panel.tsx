"use client"

import { useEffect, useState } from "react"
import { CheckCircle2, XCircle } from "lucide-react"
import { InspectionResult, RunningStats, CLASS_COLORS } from "@/types/inspection"
import { cn } from "@/lib/utils"

interface ConveyorPanelProps {
  result: InspectionResult | null
  stats:  RunningStats | null
}

export function ConveyorPanel({ result, stats }: ConveyorPanelProps) {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    setVisible(false)
    const t = setTimeout(() => setVisible(true), 120)
    return () => clearTimeout(t)
  }, [result?.index])

  const conf   = result ? Math.round(result.confidence * 100) : 0
  const colors = result ? CLASS_COLORS[result.prediction] : null

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3 h-full">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">
          Bauteil&nbsp;
          <span className="text-zinc-300">{result ? result.index.toString().padStart(4, "0") : "----"}</span>
          {result && <span className="text-zinc-600"> / {result.total}</span>}
        </span>
        {result && colors ? (
          <span className={cn(
            "inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-[10px] font-bold font-mono uppercase tracking-wider",
            colors.bg, colors.text,
          )}>
            <span className={cn("w-1.5 h-1.5 rounded-full", colors.dot)} />
            {result.prediction.replace(/_/g, " ")}
          </span>
        ) : (
          <span className="text-[10px] text-zinc-600 border border-zinc-800 rounded px-2.5 py-1 font-mono">BEREIT</span>
        )}
      </div>

      {/* ── Image ── */}
      <div className="relative flex-1 min-h-0 flex items-center justify-center bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden" style={{ minHeight: 200 }}>
        {result ? (
          <img
            src={`data:image/png;base64,${result.image_b64}`}
            alt={`Bauteil ${result.index}`}
            className={cn("w-full h-full object-contain transition-opacity duration-120",
              visible ? "opacity-100" : "opacity-0")}
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-zinc-700">
            <div className="w-12 h-12 rounded-full border border-dashed border-zinc-700 flex items-center justify-center">
              <span className="text-lg">⬡</span>
            </div>
            <span className="text-[10px] font-mono">WARTE AUF INSPEKTION</span>
          </div>
        )}
      </div>

      {/* ── Confidence bar ── */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[10px] font-mono">
          <span className="text-zinc-500">KONFIDENZ</span>
          <span className={cn("font-bold", colors?.text ?? "text-zinc-500")}>{result ? `${conf}%` : "—"}</span>
        </div>
        <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
          <div className={cn("h-full rounded-full transition-all duration-500", colors?.dot ?? "bg-zinc-700")}
            style={{ width: result ? `${conf}%` : "0%" }} />
        </div>
      </div>

      {/* ── Top-3 class probs ── */}
      <div className="space-y-1">
        {result
          ? (Object.entries(result.class_probs) as [string, number][])
              .sort((a, b) => b[1] - a[1]).slice(0, 3)
              .map(([cls, prob], i) => {
                const c = CLASS_COLORS[cls as keyof typeof CLASS_COLORS]
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <span className="text-zinc-700 font-mono text-[10px] w-3">{i + 1}</span>
                    <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", c.dot)} />
                    <span className="w-24 text-zinc-500 truncate font-mono text-[10px]">{cls.replace(/_/g, " ")}</span>
                    <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                      <div className={cn("h-full rounded-full", c.dot)} style={{ width: `${Math.round(prob * 100)}%` }} />
                    </div>
                    <span className="w-8 text-right text-zinc-600 font-mono text-[10px]">{Math.round(prob * 100)}%</span>
                  </div>
                )
              })
          : [1,2,3].map(i => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-3 text-zinc-800 font-mono text-[10px]">{i}</span>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-800" />
                <div className="w-24 h-2 bg-zinc-800 rounded animate-pulse" />
                <div className="flex-1 h-1 bg-zinc-800 rounded-full" />
                <span className="w-8 text-zinc-800 font-mono text-[10px]">—</span>
              </div>
            ))}
      </div>

      {/* ── Ground truth row ── */}
      <div className={cn(
        "flex items-center justify-between rounded px-3 py-2 text-[10px] font-mono border",
        result
          ? result.correct
            ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400"
            : "bg-red-500/10 border-red-500/30 text-red-400"
          : "border-zinc-800 text-zinc-700",
      )}>
        <span>GT: <strong>{result ? result.true_label.replace(/_/g, " ") : "—"}</strong></span>
        {result && (result.correct
          ? <span className="flex items-center gap-1"><CheckCircle2 className="h-3 w-3" /> KORREKT</span>
          : <span className="flex items-center gap-1"><XCircle className="h-3 w-3" /> FALSCH</span>)}
      </div>

      {/* ── KPI mini-strip ── */}
      <div className="border-t border-zinc-800 pt-3 grid grid-cols-4 gap-0">
        {([
          ["INSPIZIERT",  stats?.total_inspected ?? "—", "text-zinc-100"],
          ["ACCURACY",    stats ? `${(stats.accuracy * 100).toFixed(1)}%` : "—",
            stats ? (stats.accuracy > 0.90 ? "text-emerald-400" : stats.accuracy < 0.75 ? "text-red-400" : "text-amber-400") : "text-zinc-500"],
          ["Ø KONFIDENZ", stats ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—", "text-zinc-300"],
          ["FEHLER",      stats?.errors ?? "—", stats?.errors ? "text-red-400" : "text-zinc-600"],
        ] as [string, string | number, string][]).map(([label, val, cls]) => (
          <div key={label} className="flex flex-col items-center gap-0.5">
            <span className="text-[8px] text-zinc-600 font-mono uppercase tracking-wider">{label}</span>
            <span className={cn("text-sm font-bold font-mono tabular-nums", cls)}>{val}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
