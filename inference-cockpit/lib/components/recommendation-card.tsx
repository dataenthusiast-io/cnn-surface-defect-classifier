"use client"

import { AlertTriangle, Info, Wrench, Tag, Zap } from "lucide-react"
import { InspectionResult, DEFECT_INFO, PRIORITY_STYLE, CLASS_COLORS } from "@/types/inspection"
import { cn } from "@/lib/utils"

interface RecommendationCardProps {
  result: InspectionResult | null
}

export function RecommendationCard({ result }: RecommendationCardProps) {
  const info   = result ? DEFECT_INFO[result.prediction] : null
  const pStyle = info   ? PRIORITY_STYLE[info.priority]  : null
  const colors = result ? CLASS_COLORS[result.prediction] : null

  return (
    <div className={cn(
      "bg-zinc-900 rounded-xl border overflow-hidden",
      pStyle ? pStyle.border : "border-zinc-800",
    )}>
      {/* Section header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800">
        <span className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider flex items-center gap-1.5">
          <Wrench className="h-3 w-3" /> Prozessempfehlung
        </span>
        {info && pStyle && (
          <span className={cn(
            "text-[10px] font-bold font-mono px-2 py-0.5 rounded border flex items-center gap-1",
            pStyle.text, pStyle.bg, pStyle.border,
          )}>
            {info.priority === "critical"
              ? <AlertTriangle className="h-2.5 w-2.5" />
              : <Info className="h-2.5 w-2.5" />}
            {info.label}
          </span>
        )}
      </div>

      {/* Content */}
      <div className="px-4 py-3 space-y-3">
        {info && colors ? (
          <>
            {/* Defect label */}
            <div className="flex items-center gap-2">
              <span className={cn("w-2 h-2 rounded-full", colors.dot)} />
              <span className={cn("text-xs font-semibold font-mono uppercase tracking-wide", colors.text)}>
                {result!.prediction.replace(/_/g, " ")}
              </span>
            </div>

            {/* Three rows */}
            <div className="space-y-2.5">
              <Row icon={<Zap className="h-3 w-3 text-zinc-500" />} label="Ursache" value={info.cause} />
              <div className="border-t border-zinc-800" />
              <Row
                icon={<Wrench className="h-3 w-3 text-zinc-500" />}
                label="Maßnahme"
                value={info.action}
                valueClass={pStyle?.text}
              />
              <div className="border-t border-zinc-800" />
              <Row icon={<Tag className="h-3 w-3 text-zinc-500" />} label="Teileentscheid" value={info.disposition} />
            </div>
          </>
        ) : (
          <p className="text-[10px] text-zinc-600 font-mono py-2">— Inspektion starten</p>
        )}
      </div>
    </div>
  )
}

function Row({ icon, label, value, valueClass }: {
  icon: React.ReactNode
  label: string
  value: string
  valueClass?: string
}) {
  return (
    <div className="flex gap-2.5">
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div>
        <p className="text-[9px] text-zinc-600 font-mono uppercase tracking-wider mb-0.5">{label}</p>
        <p className={cn("text-xs leading-snug", valueClass ?? "text-zinc-300")}>{value}</p>
      </div>
    </div>
  )
}
