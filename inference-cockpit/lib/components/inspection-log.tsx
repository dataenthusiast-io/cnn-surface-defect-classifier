"use client"

import { InspectionResult, CLASS_COLORS, DEFECT_INFO } from "@/types/inspection"
import { cn } from "@/lib/utils"

interface InspectionLogProps {
  history: InspectionResult[]
}

export function InspectionLog({ history }: InspectionLogProps) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <span className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">Inspektionsprotokoll</span>
        <span className="text-[10px] text-zinc-600 font-mono">{history.length} Einträge</span>
      </div>

      <div className="overflow-y-auto max-h-52">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
            <tr>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600 w-12">#</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600 w-12">IMG</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600">PREDICTION</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600">PRIO</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600">KONFIDENZ</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600">GROUND TRUTH</th>
              <th className="text-left px-4 py-2 text-[10px] font-mono text-zinc-600 w-8">✓</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {history.map(r => {
              const c    = CLASS_COLORS[r.prediction]
              const info = DEFECT_INFO[r.prediction]
              return (
                <tr key={`${r.index}-${r.timestamp}`}
                  className={cn(
                    "hover:bg-zinc-800/40 transition-colors",
                    !r.correct && "bg-red-950/20",
                  )}>
                  <td className="px-4 py-2 font-mono text-zinc-600">{r.index.toString().padStart(4, "0")}</td>
                  <td className="px-4 py-2">
                    <img
                      src={`data:image/png;base64,${r.image_b64}`}
                      alt=""
                      className="w-8 h-8 rounded object-cover border border-zinc-800"
                      style={{ imageRendering: "pixelated" }}
                    />
                  </td>
                  <td className="px-4 py-2">
                    <span className={cn(
                      "inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono text-[10px] uppercase",
                      c.bg, c.text,
                    )}>
                      <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", c.dot)} />
                      {r.prediction.replace(/_/g, " ")}
                    </span>
                  </td>
                  <td className="px-4 py-2">
                    <span className={cn(
                      "text-[9px] font-mono",
                      info.priority === "critical" ? "text-rose-400"
                      : info.priority === "high"   ? "text-orange-400"
                      : info.priority === "medium" ? "text-amber-400"
                      :                              "text-zinc-600",
                    )}>
                      {info.label}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-zinc-400">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2 font-mono text-zinc-500 text-[10px]">{r.true_label.replace(/_/g, " ")}</td>
                  <td className="px-4 py-2 text-center font-mono">
                    {r.correct
                      ? <span className="text-emerald-500">✓</span>
                      : <span className="text-red-500 font-bold">✗</span>}
                  </td>
                </tr>
              )
            })}
            {history.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-zinc-600 text-xs font-mono">
                  KEINE DATEN — INSPEKTION STARTEN
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
