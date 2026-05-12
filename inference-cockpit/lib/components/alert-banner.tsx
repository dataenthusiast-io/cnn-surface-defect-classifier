"use client"

import { AlertTriangle } from "lucide-react"
import { InspectionResult, DEFECT_INFO } from "@/types/inspection"

interface AlertBannerProps {
  result: InspectionResult | null
  visible: boolean
}

export function AlertBanner({ result, visible }: AlertBannerProps) {
  if (!visible || !result) return null

  const info = DEFECT_INFO[result.prediction]

  return (
    <div className="fixed top-0 left-0 right-0 z-50 animate-slide-down">
      <div className="bg-zinc-900 border-b border-red-500/40 text-zinc-100 px-6 py-3 flex items-center gap-3 shadow-2xl">
        <div className="w-5 h-5 rounded-full bg-red-500/20 flex items-center justify-center animate-pulse-ring">
          <AlertTriangle className="h-3 w-3 text-red-400 shrink-0" />
        </div>
        <span className="font-bold text-xs font-mono text-red-400 tracking-widest uppercase">
          Falsch klassifiziert
        </span>
        <span className="text-zinc-600 text-xs">|</span>
        <span className="text-xs font-mono text-zinc-300">
          #{result.index.toString().padStart(4, "0")} ·{" "}
          Pred: <strong className="text-zinc-100">{result.prediction.replace(/_/g, " ")}</strong>
          {" "}≠ True: <strong className="text-zinc-100">{result.true_label.replace(/_/g, " ")}</strong>
        </span>
        <span className="ml-auto text-[10px] font-mono text-zinc-500">
          {info.action}
        </span>
      </div>
    </div>
  )
}
