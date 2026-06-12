"use client"

import { AlertTriangle } from "lucide-react"
import { RunningStats, CLASS_COLORS, DefectClass, DEFECT_INFO } from "@/types/inspection"
import { cn } from "@/lib/utils"

interface StatsPanelProps {
  stats: RunningStats | null
}

export function StatsPanel({ stats }: StatsPanelProps) {
  const total       = stats?.total_inspected ?? 0
  const classCounts = stats?.class_counts    ?? {}

  // Trend alert: any class exceeds 40% of inspected
  const dominant = total > 10
    ? (Object.entries(classCounts) as [DefectClass, number][]).find(([, n]) => n / total > 0.40)
    : undefined

  return (
    <div className="flex flex-col gap-3">

      {/* Trend alert */}
      {dominant && (() => {
        const [cls, cnt] = dominant
        const info  = DEFECT_INFO[cls]
        const c     = CLASS_COLORS[cls]
        return (
          <div className="border border-amber-500/25 bg-amber-500/5 rounded-xl px-4 py-3 flex items-start gap-3">
            <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-[10px] font-bold text-amber-400 font-mono uppercase tracking-wide mb-0.5">
                Prozessdrift erkannt
              </p>
              <p className="text-xs text-zinc-300">
                <span className={cn("font-semibold", c.text)}>{cls.replace(/_/g, " ")}</span>
                {" "}·{" "}{Math.round((cnt / total) * 100)}% der Predictions.
                {" "}<span className="text-zinc-400">{info.action}</span>
              </p>
            </div>
          </div>
        )
      })()}

      {/* Class distribution */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3">
        <p className="text-[9px] text-zinc-600 font-mono uppercase tracking-widest mb-3">Klassenverteilung · Predictions</p>
        <div className="space-y-2">
          {total === 0 ? (
            [0,1,2,3].map(i => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-800" />
                <div className="w-28 h-2 bg-zinc-800 rounded animate-pulse" />
                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full" />
                <span className="w-4 text-zinc-800 font-mono text-[10px]">0</span>
              </div>
            ))
          ) : (
            (Object.entries(classCounts) as [DefectClass, number][])
              .sort((a, b) => b[1] - a[1])
              .map(([cls, count]) => {
                const pct   = (count / total) * 100
                const c     = CLASS_COLORS[cls]
                const info  = DEFECT_INFO[cls]
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", c.dot)} />
                    <span className="w-28 text-zinc-400 truncate font-mono text-[10px]">{cls.replace(/_/g, " ")}</span>
                    <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                      <div className={cn("h-full rounded-full transition-all", c.dot)} style={{ width: `${pct}%` }} />
                    </div>
                    <span className="w-5 text-right text-zinc-500 font-mono text-[10px]">{count}</span>
                    <span className={cn(
                      "text-[9px] font-mono px-1.5 py-px rounded border w-14 text-center",
                      info.priority === "critical" ? "text-rose-400   border-rose-500/30   bg-rose-500/5"
                      : info.priority === "high"   ? "text-orange-400 border-orange-500/30 bg-orange-500/5"
                      : info.priority === "medium" ? "text-amber-400  border-amber-500/30  bg-amber-500/5"
                      :                              "text-zinc-500   border-zinc-700       bg-zinc-800",
                    )}>
                      {info.label}
                    </span>
                  </div>
                )
              })
          )}
        </div>
      </div>
    </div>
  )
}
