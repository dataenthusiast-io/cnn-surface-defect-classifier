export type DefectClass =
  | "crazing"
  | "inclusion"
  | "patches"
  | "pitted_surface"
  | "rolled-in_scale"
  | "scratches"

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
  "crazing":         { ring: "ring-amber-500",   text: "text-amber-400",   dot: "bg-amber-500",   bg: "bg-amber-500/10"   },
  "inclusion":       { ring: "ring-rose-500",    text: "text-rose-400",    dot: "bg-rose-500",    bg: "bg-rose-500/10"    },
  "patches":         { ring: "ring-zinc-400",    text: "text-zinc-300",    dot: "bg-zinc-400",    bg: "bg-zinc-700/40"    },
  "pitted_surface":  { ring: "ring-orange-500",  text: "text-orange-400",  dot: "bg-orange-500",  bg: "bg-orange-500/10"  },
  "rolled-in_scale": { ring: "ring-cyan-500",    text: "text-cyan-400",    dot: "bg-cyan-500",    bg: "bg-cyan-500/10"    },
  "scratches":       { ring: "ring-red-500",     text: "text-red-400",     dot: "bg-red-500",     bg: "bg-red-500/10"     },
}

export const CLASS_HEX: Record<DefectClass, string> = {
  "crazing":         "#f59e0b",
  "inclusion":       "#f43f5e",
  "patches":         "#a1a1aa",
  "pitted_surface":  "#f97316",
  "rolled-in_scale": "#06b6d4",
  "scratches":       "#ef4444",
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
  "inclusion": {
    priority:    "critical",
    label:       "KRITISCH",
    cause:       "Materialfehler im Rohstoff (Schlackeneinschluss)",
    action:      "Produktionslinie stoppen — Rohmaterialcharge prüfen und sperren",
    disposition: "Ausschuss — nicht weiterverwendbar",
  },
  "scratches": {
    priority:    "high",
    label:       "HOCH",
    cause:       "Mechanische Beschädigung in der Walzlinie",
    action:      "Werkzeugverschleiß messen, Führungsrollen & Kaliber kontrollieren",
    disposition: "Ausschuss bei Tiefe > Toleranz",
  },
  "pitted_surface": {
    priority:    "high",
    label:       "HOCH",
    cause:       "Kühlmittel- oder Schmierungsproblem, Korrosionsanzeichen",
    action:      "Kühlsystem & Schmierung überprüfen, Lagerungsbedingungen kontrollieren",
    disposition: "Ausschuss — Nachfolgecharge beobachten",
  },
  "rolled-in_scale": {
    priority:    "high",
    label:       "HOCH",
    cause:       "Zunder eingewalzt — Entzunderungsanlage unterlastet",
    action:      "Entzunderungsanlage prüfen, Vorwärmtemperatur und Haspeldruck anpassen",
    disposition: "Ausschuss",
  },
  "crazing": {
    priority:    "medium",
    label:       "MITTEL",
    cause:       "Materialermüdung / thermische Eigenspannungen",
    action:      "Walzendruck & Temperaturprofil anpassen, Abkühlrate verringern",
    disposition: "Prüfung erforderlich — je nach Tiefe der Risse",
  },
  "patches": {
    priority:    "low",
    label:       "GERING",
    cause:       "Oberflächliche Unregelmäßigkeit, leichte Oxidation",
    action:      "Sichtprüfung — Kundenvorgaben für Oberflächenklasse prüfen",
    disposition: "Eventuell für Abnehmer B/C verwendbar",
  },
}

export const PRIORITY_STYLE: Record<Priority, { text: string; bg: string; border: string }> = {
  critical: { text: "text-rose-400",   bg: "bg-rose-500/10",   border: "border-rose-500/40"   },
  high:     { text: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/40" },
  medium:   { text: "text-amber-400",  bg: "bg-amber-500/10",  border: "border-amber-500/40"  },
  low:      { text: "text-zinc-400",   bg: "bg-zinc-700/30",   border: "border-zinc-600/40"   },
}
