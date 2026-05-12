"use client"

import { Play, Pause, RotateCcw, Wifi, WifiOff } from "lucide-react"
import { Button } from "@/lib/components/ui/button"

interface ControlBarProps {
  isRunning:  boolean
  speed:      number
  connected:  boolean
  onToggle:   () => void
  onReset:    () => void
  onSpeed:    (ms: number) => void
}

export function ControlBar({ isRunning, speed, connected, onToggle, onReset, onSpeed }: ControlBarProps) {
  const speedLabel = speed <= 800 ? "Sehr schnell" : speed <= 1500 ? "Schnell" : speed <= 2500 ? "Mittel" : speed <= 3500 ? "Langsam" : "Sehr langsam"

  return (
    <div className="flex items-center gap-4 bg-zinc-900 border border-zinc-800 px-5 py-2.5 rounded-xl">
      <Button
        onClick={onToggle}
        size="sm"
        className={isRunning
          ? "bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border border-amber-500/40 min-w-[96px]"
          : "bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/40 min-w-[96px]"}
      >
        {isRunning
          ? <><Pause className="h-3.5 w-3.5 mr-1.5" />Pause</>
          : <><Play  className="h-3.5 w-3.5 mr-1.5" />Start</>}
      </Button>

      <Button onClick={onReset} size="sm"
        className="bg-transparent hover:bg-zinc-800 text-zinc-400 border border-zinc-700 hover:text-zinc-200">
        <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
        Reset
      </Button>

      <div className="w-px h-5 bg-zinc-800" />

      <div className="flex items-center gap-3 flex-1 max-w-xs">
        <span className="text-xs text-zinc-500 whitespace-nowrap">Durchsatz</span>
        <span className="text-xs text-zinc-600">↑</span>
        <input
          type="range"
          min={500} max={5000} step={100}
          value={speed}
          onChange={e => onSpeed(Number(e.target.value))}
          className="flex-1 accent-zinc-400 cursor-pointer h-1"
        />
        <span className="text-xs text-zinc-600">↓</span>
        <span className="text-xs text-zinc-400 w-20 text-right font-mono">
          {speedLabel}
        </span>
      </div>

      <div className="w-px h-5 bg-zinc-800" />

      <div className={`flex items-center gap-1.5 text-xs ml-auto font-mono ${connected ? "text-emerald-500" : "text-red-500"}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"} ${connected && isRunning ? "animate-pulse" : ""}`} />
        {connected ? "API verbunden" : "Offline"}
      </div>
    </div>
  )
}
