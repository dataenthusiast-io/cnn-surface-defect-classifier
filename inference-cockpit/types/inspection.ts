export type DefectClass =
  | "Abdruck 1"
  | "Abdruck 2"
  | "Stanzfehler"
  | "i.O.-Teile"

export interface InspectionResult {
  index:       number
  total:       number
  image_b64:   string
  true_label:  DefectClass
  prediction:  DefectClass
  confidence:  number
  correct:     boolean
  class_probs: Record<DefectClass, number>
  timestamp:   number
}

export interface RunningStats {
  total_inspected: number
  accuracy:        number
  errors:          number
  avg_confidence:  number
  class_counts:    Record<DefectClass, number>
}

// ─── Visual palette ────────────────────────────────────────────────────────
export const CLASS_COLORS: Record<DefectClass, { ring: string; text: string; dot: string; bg: string }> = {
  "Abdruck 1":  { ring: "ring-amber-500",   text: "text-amber-400",   dot: "bg-amber-500",   bg: "bg-amber-500/10"   },
  "Abdruck 2":  { ring: "ring-orange-500",  text: "text-orange-400",  dot: "bg-orange-500",  bg: "bg-orange-500/10"  },
  "Stanzfehler":{ ring: "ring-rose-500",    text: "text-rose-400",    dot: "bg-rose-500",    bg: "bg-rose-500/10"    },
  "i.O.-Teile": { ring: "ring-emerald-500", text: "text-emerald-400", dot: "bg-emerald-500", bg: "bg-emerald-500/10" },
}

export const CLASS_HEX: Record<DefectClass, string> = {
  "Abdruck 1":  "#f59e0b",
  "Abdruck 2":  "#f97316",
  "Stanzfehler":"#f43f5e",
  "i.O.-Teile": "#10b981",
}

// ─── Business logic ────────────────────────────────────────────────────────
export type Priority = "critical" | "high" | "medium" | "low"

export interface DefectInfo {
  priority:    Priority
  label:       string          // Short German priority label
  cause:       string          // Root cause
  action:      string          // Recommended process action
  disposition: string          // Part disposition
}

export const DEFECT_INFO: Record<DefectClass, DefectInfo> = {
  "Stanzfehler": {
    priority:    "critical",
    label:       "KRITISCH",
    cause:       "Stanzprozess fehlerhaft — Werkzeugbruch oder Materialversagen",
    action:      "Produktionslinie stoppen — Stanzwerkzeug prüfen und sperren",
    disposition: "Ausschuss — nicht weiterverwendbar",
  },
  "Abdruck 2": {
    priority:    "high",
    label:       "HOCH",
    cause:       "Werkzeugabdruck Typ 2 — erhöhter Werkzeugverschleiß",
    action:      "Werkzeugverschleiß messen, Oberfläche und Führungen kontrollieren",
    disposition: "Ausschuss",
  },
  "Abdruck 1": {
    priority:    "medium",
    label:       "MITTEL",
    cause:       "Werkzeugabdruck Typ 1 — leichte Oberflächenmarkierung",
    action:      "Werkzeugoberfläche prüfen, Schmiermittelauftrag kontrollieren",
    disposition: "Nacharbeit möglich — Sichtprüfung erforderlich",
  },
  "i.O.-Teile": {
    priority:    "low",
    label:       "GERING",
    cause:       "Kein Defekt erkannt",
    action:      "Keine Maßnahme erforderlich",
    disposition: "Freigabe",
  },
}

export const PRIORITY_STYLE: Record<Priority, { text: string; bg: string; border: string }> = {
  critical: { text: "text-rose-400",   bg: "bg-rose-500/10",   border: "border-rose-500/40"   },
  high:     { text: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/40" },
  medium:   { text: "text-amber-400",  bg: "bg-amber-500/10",  border: "border-amber-500/40"  },
  low:      { text: "text-zinc-400",   bg: "bg-zinc-700/30",   border: "border-zinc-600/40"   },
}
