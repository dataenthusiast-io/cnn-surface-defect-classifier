"use client"

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Dot,
} from "recharts"
import { InspectionResult, CLASS_HEX, DefectClass } from "@/types/inspection"

interface ConfidenceChartProps {
  history: InspectionResult[]
}

export function ConfidenceChart({ history }: ConfidenceChartProps) {
  const data = history
    .slice(0, 24)
    .reverse()
    .map(r => ({
      index:      r.index,
      confidence: Math.round(r.confidence * 1000) / 10,
      prediction: r.prediction,
      correct:    r.correct,
      color:      CLASS_HEX[r.prediction],
    }))

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3">
      <p className="text-[10px] text-zinc-500 font-mono uppercase tracking-wider">Konfidenz-Verlauf · Letzte 24</p>

      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#27272a" />
          <XAxis
            dataKey="index"
            tick={{ fontSize: 9, fill: "#52525b", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 9, fill: "#52525b", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "monospace",
              color: "#d4d4d8",
            }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={((v: any, _: any, entry: any) => [
              `${v}%  ${entry.payload.correct ? "✓" : "✗"}`,
              (entry.payload.prediction as DefectClass).replace(/_/g, " "),
            ]) as any}
            labelFormatter={(l: any) => `#${l}`}
          />
          <ReferenceLine
            y={80}
            stroke="#3f3f46"
            strokeDasharray="4 4"
            label={{ value: "80%", fontSize: 9, fill: "#52525b", fontFamily: "monospace" }}
          />
          <Line
            type="monotone"
            dataKey="confidence"
            stroke="#3f3f46"
            strokeWidth={1}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            dot={((props: any) => (
              <Dot
                cx={props.cx}
                cy={props.cy}
                r={3.5}
                fill={props.payload.color}
                stroke={props.payload.correct ? "#09090b" : "#fafafa"}
                strokeWidth={props.payload.correct ? 1 : 1.5}
              />
            )) as any}
            activeDot={{ r: 5, stroke: "#71717a" }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {(Object.entries(CLASS_HEX) as [DefectClass, string][]).map(([cls, hex]) => (
          <span key={cls} className="flex items-center gap-1 text-[10px] text-zinc-500 font-mono">
            <span className="w-2 h-2 rounded-full inline-block shrink-0" style={{ background: hex }} />
            {cls.replace(/_/g, " ")}
          </span>
        ))}
        <span className="flex items-center gap-1 text-[10px] text-zinc-600 ml-auto">
          ◯ weißer Rand = korrekt · dunkler Rand = Fehler
        </span>
      </div>
    </div>
  )
}
