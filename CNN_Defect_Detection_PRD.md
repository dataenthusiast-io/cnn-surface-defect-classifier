# PRD: CNN-basierter Defektdetektor – Industrielles Inference-Cockpit

**Projekt:** FOM Transfer-Projekt – Datenanalyse in der Industrie  
**Team:** Burak Yilmaz (Business), Sean O'Brien (Engineering), Nicolas (Architektur)  
**Stack:** PyTorch · FastAPI · Next.js · shadcn/ui  
**Ziel:** Trainiertes CNN-Modell + lokales Echtzeit-Inferenz-Cockpit das eine fiktive Produktionslinie simuliert

---

## Gesamtarchitektur

```
┌─────────────────────────────────────────────────────┐
│              TEIL A: ML + BACKEND                   │
│                                                     │
│  NEU Dataset → Preprocessing → ResNet-18 Training  │
│       → best_model.pth → FastAPI Inference Server  │
└────────────────────────┬────────────────────────────┘
                         │ HTTP Polling (GET /predict/next)
┌────────────────────────▼────────────────────────────┐
│              TEIL B: FRONTEND COCKPIT               │
│                                                     │
│     Next.js + shadcn/ui – läuft lokal auf :3000    │
│     Simuliert fiktive Produktionslinie in Echtzeit │
└─────────────────────────────────────────────────────┘
```

Beide Teile laufen vollständig lokal. Kein Cloud-Deployment, keine Datenpersistenz, keine Authentifizierung.

---

# TEIL A: ML-Modell, Training & Backend

---

## A1. Technische Entscheidungsbegründungen

### A1.1 Warum ResNet-18?

ResNet-18 ist die kleinste Variante der ResNet-Familie (He et al., 2016) mit ~11 Millionen Parametern. Sie führt **Residualverbindungen** ein: statt nur `F(x)` zu lernen, lernt das Netz `F(x) + x`. Das verhindert das Verschwinden von Gradienten in tiefen Netzen. Für diesen PoC entscheidend: ResNet-18 ist klein genug für lokales Training auf M2 (< 5 Min./Epoch auf MPS) und groß genug, um Oberflächentexturen zuverlässig zu unterscheiden. ResNet-50 oder EfficientNet wären für 1.800 Bilder überdimensioniert.

### A1.2 Warum Transfer Learning – und welche Schichten einfrieren?

Ein auf ImageNet vortrainiertes ResNet-18 hat bereits gelernt, visuelle Primitive zu erkennen. Diese Merkmale sind domänenunabhängig und auf Industriebilder übertragbar.

**ResNet-18-Struktur:**

```
Stem        → conv1, bn1, maxpool       generisch: Kanten, Helligkeit
layer1      → 2x Residualblock          einfache Texturen, Ecken
layer2      → 2x Residualblock          komplexere Texturen, Muster
layer3      → 2x Residualblock          objektspezifische Merkmale
layer4      → 2x Residualblock          hochsemantische Merkmale
avgpool+fc  → Klassifikations-Head      ImageNet-spezifisch → wird ersetzt
```

**2-Phasen-Strategie:**

- **Phase 1 (5 Epochs) – nur Head:** Backbone komplett eingefroren. Bringt den zufällig initialisierten Head auf sinnvolle Ausgangswerte, bevor tiefere Schichten aufgetaut werden. Verhindert dass große Gradienten vom untrainierten Head die vortrainierten Gewichte zerstören.
- **Phase 2 (20 Epochs) – Head + layer3 + layer4:** Stem + layer1 + layer2 bleiben eingefroren (generische Primitiven, kein Anpassungsbedarf). layer3/4 werden mit niedrigerer LR fine-getuned – diese Schichten müssen von ImageNet-Semantik auf industrielle Oberflächendefekte umtrainiert werden. Der Head trainiert weiter mit höherer LR.

Alles einfrieren wäre "Feature Extraction" – für industrielle Texturen nicht ausreichend. Alles auftauen würde mit 1.260 Trainingsbildern sofort Overfitting produzieren.

### A1.3 Warum MPS statt CPU?

PyTorchs MPS-Backend nutzt den Neural Engine und GPU-Kerne des M2-Chips. Für ResNet-18 auf 200×200-Bildern ca. 4-6x Speedup gegenüber CPU. Aktivierung: `device = torch.device("mps")`. Kein Code-Umbau nötig.

### A1.4 Warum binäre Klassifikation?

Der NEU-Datensatz hat 6 Klassen. Die primäre Entscheidung auf einer Produktionslinie ist binär: i.O. oder n.i.O. Multi-Class würde den PoC-Scope sprengen und die Kommunikation gegenüber dem Business-Stakeholder erschweren. Welche Klassen zu "Defekt" gebündelt werden, ist Teil von Buraks Use-Case-Definition.

### A1.5 Warum FastAPI + Polling statt WebSocket?

Die Simulation ist unidirektional und taktgesteuert durch einen Slider im Frontend – der Client bestimmt das Intervall selbst. Polling ist hier ausreichend und deutlich einfacher: der Frontend-Loop ruft alle N Millisekunden `GET /predict/next` auf und updated den React-State. WebSocket oder SSE wären Overengineering für einen lokalen Demo-Kontext ohne echten asynchronen Push-Bedarf.

---

## A2. Projektstruktur

```
cnn-defect-detector/
│
├── data/
│   ├── raw/                    # NEU-Datensatz, unverändert (ein Ordner pro Klasse)
│   └── processed/              # gecachte Splits (optional)
│
├── src/
│   ├── dataset.py              # PyTorch Dataset, Augmentierung, Splits
│   ├── model.py                # ResNet-18, Layer-Freeze-Logik, Head-Ersatz
│   ├── train.py                # Trainingsloop, Logging, Early Stopping
│   ├── evaluate.py             # Metriken, Plots, Fehleranalyse
│   └── predict.py              # Einzelbild-Inferenz CLI
│
├── api/
│   ├── main.py                 # FastAPI App, Routen
│   └── pipeline.py             # InferencePipeline-Klasse, Testset-Iterator
│
├── outputs/
│   ├── checkpoints/            # best_model.pth, last_model.pth
│   ├── plots/                  # Trainingsplots (PNG)
│   └── logs/                   # metrics.csv
│
├── config.py
├── requirements.txt
└── README.md
```

---

## A3. Konfiguration (`config.py`)

```python
from pathlib import Path

DATA_DIR        = Path("data/raw")
OUTPUT_DIR      = Path("outputs")
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
PLOT_DIR        = OUTPUT_DIR / "plots"
LOG_PATH        = OUTPUT_DIR / "logs" / "metrics.csv"

IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 4
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# Klassen-Mapping – Burak definiert dies anhand des Use Case
# NEU-Klassen: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
OK_CLASSES      = ["patches"]
DEFECT_CLASSES  = ["crazing", "scratches"]
CLASS_NAMES     = ["i.O.", "n.i.O."]

PHASE1_EPOCHS       = 5
PHASE1_LR_HEAD      = 1e-3

PHASE2_EPOCHS       = 20
PHASE2_LR_HEAD      = 1e-3
PHASE2_LR_BACKBONE  = 1e-4

PATIENCE            = 5

DEVICE              = "mps"     # Fallback: "cuda" oder "cpu"

API_HOST            = "127.0.0.1"
API_PORT            = 8000
```

---

## A4. Datenpipeline (`src/dataset.py`)

- Liest NEU-Datensatz aus `data/raw/` (ein Ordner pro Klasse)
- Mappt Klassen auf binär gemäß `config.OK_CLASSES` / `config.DEFECT_CLASSES`; nicht gelistete Klassen werden übersprungen
- Stratifizierter Split 70/15/15 mit fixem `RANDOM_SEED`
- Separate Transforms für Train vs. Val/Test

```python
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Begründung ImageNet-Normalisierung:** Das vortrainierte ResNet-18 wurde mit diesen Statistiken trainiert. Abweichende Normalisierung degradiert die Merkmalextraktion der eingefrorenen Schichten messbar.

**Interface:**

```python
class NEUDefectDataset(Dataset):
    def __init__(self, root_dir, split, transform=None): ...
    def __len__(self): ...
    def __getitem__(self, idx) -> tuple[Tensor, int]: ...

def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    # gibt (train_loader, val_loader, test_loader) zurück
```

---

## A5. Modell (`src/model.py`)

```python
def build_model(device: str) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model.to(device)

def freeze_for_phase1(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_for_phase2(model: nn.Module):
    for name, param in model.named_parameters():
        if any(name.startswith(l) for l in ["layer3", "layer4", "fc"]):
            param.requires_grad = True

def get_optimizer_phase1(model) -> optim.Optimizer:
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR_HEAD
    )

def get_optimizer_phase2(model) -> optim.Optimizer:
    return optim.Adam([
        {"params": model.layer3.parameters(), "lr": PHASE2_LR_BACKBONE},
        {"params": model.layer4.parameters(), "lr": PHASE2_LR_BACKBONE},
        {"params": model.fc.parameters(),     "lr": PHASE2_LR_HEAD},
    ])
```

**Begründung Dropout(0.3):** Mit ~1.260 Trainingsbildern ist Regularisierung im Head kritisch. Dropout deaktiviert zufällig 30% der Neuronen pro Forward-Pass und erzwingt robustere Repräsentationen.

---

## A6. Trainingsloop (`src/train.py`)

- Phase 1 und Phase 2 sequenziell
- Logging nach jeder Epoch nach `outputs/logs/metrics.csv`
- Felder: `epoch, phase, train_loss, val_loss, train_acc, val_acc, val_f1, lr, duration_s`
- Early Stopping auf `val_f1`, Patience = `PATIENCE`
- Speichert `best_model.pth` (bestes Val-F1) und `last_model.pth`
- CosineAnnealingLR in Phase 2

**Konsolenausgabe pro Epoch:**
```
Epoch 07/25 | Phase 2 | Loss 0.312→0.287 | Acc 86.4→88.2% | F1 0.879 | 4.2s
```

---

## A7. Evaluation & Visualisierungen (`src/evaluate.py`)

Lädt `best_model.pth`, evaluiert auf Testset, generiert alle Plots nach `outputs/plots/`.

### Metriken-Output
```
Test-Ergebnisse
──────────────────────────────────
Accuracy:     91.3%
Macro F1:     0.908
──────────────────────────────────
             precision  recall  f1
i.O.           0.93      0.91   0.92
n.i.O.         0.90      0.92   0.91
──────────────────────────────────
False Negatives (verpasste Defekte): 6
```

### Plot 1: Trainingskurven (`training_curves.png`)
2×2 Grid, wird während Training nach jeder Epoch überschrieben:
- Train vs. Val Loss
- Train vs. Val Accuracy
- Val F1 + Best-F1-Marker
- Learning Rate pro Epoch

Vertikale gestrichelte Linie markiert Übergang Phase 1 → Phase 2.

### Plot 2: Confusion Matrix (`confusion_matrix.png`)
Normalisierte Heatmap auf Testset. Explizite Hervorhebung der False-Negative-Rate.

### Plot 3: Beispielvorhersagen (`sample_predictions.png`)
4×4 Grid zufälliger Testbilder. Grüner Rahmen = korrekt, roter Rahmen = falsch.  
Titel pro Bild: `True: n.i.O. | Pred: i.O. | Conf: 67%`

### Plot 4: Fehlerbilder (`false_negatives.png`, `false_positives.png`)
Alle Fehler aus dem Testset in separaten Grids. Wichtigstes qualitatives Deliverable – zeigt sofort welche Defektmuster das Modell nicht erkennt.

### Plot 5: Konfidenzverteilung (`confidence_distribution.png`)
Histogramm der Softmax-Konfidenzwerte, getrennt nach korrekt/falsch klassifiziert. Zeigt ob das Modell bei Fehlern unsicher ist (gewünscht) oder mit hoher Konfidenz falsch liegt (problematisch).

---

## A8. Inference API (`api/`)

### `api/pipeline.py` – InferencePipeline

```python
class InferencePipeline:
    """
    Lädt best_model.pth einmalig beim Start.
    Hält internen Index der durch den Testset iteriert.
    Liefert bei jedem next()-Aufruf Bild (base64) + Prediction + Statistiken.
    """
    def __init__(self, model_path: str, test_dataset: NEUDefectDataset): ...

    def next(self) -> dict:
        # index++ mit Wraparound am Testset-Ende
        return {
            "index":       int,
            "total":       int,
            "image_b64":   str,          # base64-enkodiertes PNG
            "true_label":  str,          # "i.O." oder "n.i.O."
            "prediction":  str,
            "confidence":  float,        # 0.0 – 1.0
            "correct":     bool,
            "class_probs": {
                "i.O.":   float,
                "n.i.O.": float
            }
        }

    def reset(self): ...

    def get_stats(self) -> dict:
        return {
            "total_inspected": int,
            "defect_rate":     float,
            "accuracy":        float,
            "avg_confidence":  float,
            "false_negatives": int,
            "false_positives": int
        }
```

**Laufende Statistiken:** Da Ground-Truth-Labels bekannt sind, können Accuracy und Fehlertypen live berechnet werden. Man sieht in der Demo wie sich die laufende Accuracy über Zeit stabilisiert – pädagogisch wertvolles Detail.

### `api/main.py` – FastAPI Routen

```python
app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"], allow_methods=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model": "best_model.pth", "device": DEVICE}

@app.get("/predict/next")
def predict_next():
    return pipeline.next()

@app.get("/predict/reset")
def predict_reset():
    pipeline.reset()
    return {"status": "reset"}

@app.get("/stats")
def get_stats():
    return pipeline.get_stats()
```

### Start Backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## A9. Erfolgskriterien ML

| Metrik | Zielwert |
|---|---|
| Test Accuracy | > 88% |
| Macro F1 | > 0.80 |
| False Negative Rate | < 15% |
| Trainingszeit gesamt | < 15 Min. auf M2 MPS |

---

# TEIL B: Frontend – Inference Cockpit

---

## B1. Konzept & Designziel

Das Cockpit simuliert ein fiktives industrielles Qualitätssicherungs-Dashboard. Eine virtuelle Produktionslinie liefert Bauteilbilder, das trainierte CNN klassifiziert sie in Echtzeit. Ziel ist eine Demo-Oberfläche die in einer Präsentation ohne Erklärung sofort verständlich ist.

**Kernprinzipien:**
- Klar, professionell, nicht verspielt
- i.O. = grün, n.i.O. = rot – konsequent durchgehalten
- Defekt-Alarm visuell dominant
- Liniendurchsatz per Slider steuerbar
- Kein Speichern, kein Reload-Persist – reine Session-State

---

## B2. Tech Stack

```
Next.js 14 (App Router)
shadcn/ui – Card, Badge, Button, Slider, Progress, Separator
Tailwind CSS
Recharts – Konfidenz-Liniendiagramm
Lucide Icons
```

---

## B3. Projektstruktur Frontend

```
cockpit/
├── app/
│   ├── layout.tsx
│   └── page.tsx                  # Root – rendert <CockpitDashboard />
│
├── components/
│   ├── cockpit-dashboard.tsx     # Hauptkomponente, hält gesamten State
│   ├── conveyor-panel.tsx        # Aktuelles Bild + Prediction
│   ├── stats-panel.tsx           # 4 KPI-Karten
│   ├── confidence-chart.tsx      # Recharts Liniendiagramm letzte N Werte
│   ├── inspection-log.tsx        # Scrollbare Tabelle letzter Inspektionen
│   ├── alert-banner.tsx          # Roter Alert bei n.i.O.-Prediction
│   └── control-bar.tsx           # Play/Pause, Speed-Slider, Reset
│
├── lib/
│   └── api.ts                    # fetch-Wrapper für Backend-Routen
│
└── types/
    └── inspection.ts             # TypeScript-Interfaces
```

---

## B4. TypeScript Interfaces (`types/inspection.ts`)

```typescript
interface InspectionResult {
  index:       number
  total:       number
  image_b64:   string
  true_label:  "i.O." | "n.i.O."
  prediction:  "i.O." | "n.i.O."
  confidence:  number              // 0.0 – 1.0
  correct:     boolean
  class_probs: { "i.O.": number; "n.i.O.": number }
  timestamp:   number              // Date.now(), clientseitig gesetzt
}

interface RunningStats {
  total_inspected: number
  defect_rate:     number
  accuracy:        number
  avg_confidence:  number
  false_negatives: number
  false_positives: number
}
```

---

## B5. State Management (`cockpit-dashboard.tsx`)

Alles in einer Komponente via `useState` / `useRef`. Kein externer Store nötig.

```typescript
const [isRunning, setIsRunning] = useState(false)
const [speed, setSpeed]         = useState(2000)     // ms zwischen Polls
const [current, setCurrent]     = useState<InspectionResult | null>(null)
const [history, setHistory]     = useState<InspectionResult[]>([])
const [stats, setStats]         = useState<RunningStats | null>(null)
const [isAlert, setIsAlert]     = useState(false)
const intervalRef               = useRef<NodeJS.Timeout>()
```

**Polling-Loop:**

```typescript
useEffect(() => {
  if (!isRunning) return
  intervalRef.current = setInterval(async () => {
    const result: InspectionResult = {
      ...(await api.predictNext()),
      timestamp: Date.now()
    }
    setCurrent(result)
    setHistory(prev => [result, ...prev].slice(0, 50))
    if (result.prediction === "n.i.O.") triggerAlert()
    setStats(await api.getStats())
  }, speed)
  return () => clearInterval(intervalRef.current)
}, [isRunning, speed])
```

---

## B6. Komponenten-Spec

### `control-bar.tsx`

Leiste oben über dem gesamten Dashboard:

- **Play/Pause Button** (Lucide `Play` / `Pause` Icon)
- **Speed Slider** (shadcn Slider, 500ms – 5000ms, invertiertes Label: "Liniendurchsatz: schnell ↔ langsam")
- **Reset Button** (ruft `api.reset()` auf, leert `history` + `stats` im State)
- **Verbindungsstatus** (grüner Dot + "Verbunden" wenn `/health` antwortet, rot + "Offline" sonst)

---

### `conveyor-panel.tsx`

Zentrale Karte. Das Herzstück der Demo.

```
┌──────────────────────────────────────────────┐
│  BAUTEIL #047                [ n.i.O. ]      │
│                          Badge: rot/grün     │
│  ┌────────────────────────────────────────┐  │
│  │                                        │  │
│  │         Bauteilbild (base64 img)       │  │
│  │         224×224, zentriert             │  │
│  │                                        │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Konfidenz: 94.2%                            │
│  [████████████████████░░░░]  Progress Bar   │
│                                              │
│  Ground Truth: n.i.O.   ✓ Korrekt           │
└──────────────────────────────────────────────┘
```

- Badge: `variant="destructive"` bei n.i.O., eigenes grünes Badge bei i.O.
- Progress Bar: Konfidenzwert, Farbe korrespondiert mit Prediction
- Ground Truth Zeile: immer sichtbar; bei falscher Klassifikation fett rot hervorgehoben
- Bildwechsel: kurze CSS Opacity-Transition (150ms) – kein hartes Snap

---

### `alert-banner.tsx`

Erscheint für 2s nach jeder n.i.O.-Prediction, auto-dismiss danach.

```
┌──────────────────────────────────────────────────┐
│  ⚠  DEFEKT ERKANNT  –  Bauteil #047             │
│     Konfidenz: 94.2%  |  Ausschleusen empfohlen  │
└──────────────────────────────────────────────────┘
```

Roter Hintergrund, weiße Schrift, slide-in von oben. Kein Modal, kein Blocking. Implementiert via `useState(false)` + `setTimeout(() => setIsAlert(false), 2000)`.

---

### `stats-panel.tsx`

4 KPI-Karten in einer Reihe:

| Karte | Wert | Farblogik |
|---|---|---|
| Inspiziert | `total_inspected` | neutral |
| Defektrate | `defect_rate %` | > 30% rot, < 10% grün, sonst neutral |
| Modell-Accuracy | `accuracy %` | > 85% grün, < 70% rot, sonst neutral |
| Ø Konfidenz | `avg_confidence %` | neutral |

Farbkodierung über Tailwind conditional classes, nicht inline styles.

---

### `confidence-chart.tsx`

Recharts `LineChart`, letzte 20 Inspektionen:

- X-Achse: Bauteil-Index (oder laufende Nummer)
- Y-Achse: Konfidenz 0–100%
- Datenpunkte: grün wenn `prediction === "i.O."`, rot wenn n.i.O.
- Referenzlinie bei 80% als gestrichelter Schwellwert
- Tooltip zeigt: Index, Prediction, Konfidenz

---

### `inspection-log.tsx`

Scrollbare Tabelle der letzten 50 Inspektionen, neueste oben:

| # | Vorschau | Prediction | Konfidenz | Ground Truth | Status |
|---|---|---|---|---|---|
| 047 | 40×40 Thumbnail | n.i.O. 🔴 | 94.2% | n.i.O. | ✓ |
| 046 | 40×40 Thumbnail | i.O. 🟢 | 91.7% | i.O. | ✓ |
| 045 | 40×40 Thumbnail | i.O. 🟢 | 73.1% | n.i.O. | ✗ |

- Falsch klassifizierte Zeilen: leicht roter Zeilenhintergrund (`bg-red-50`)
- Tabelle scrollt intern (`overflow-y-auto max-h-64`), nicht die ganze Seite

---

## B7. Layout (Desktop, 1280px+)

```
┌──────────────────────────────────────────────────────────┐
│         CONTROL BAR  (Play | Slider | Reset | Status)   │
├─────────────────────────┬────────────────────────────────┤
│                         │                                │
│    CONVEYOR PANEL       │     STATS PANEL (4 KPIs)      │
│    (aktuelles Bild)     │                                │
│                         ├────────────────────────────────┤
│                         │                                │
│                         │     CONFIDENCE CHART           │
│                         │                                │
├─────────────────────────┴────────────────────────────────┤
│                  INSPECTION LOG (scrollbar)              │
└──────────────────────────────────────────────────────────┘

ALERT BANNER floatet oben, position: absolute, z-50
```

Tailwind Grid: `grid-cols-[400px_1fr]` für linke/rechte Spalte. Conveyor Panel links, Stats + Chart rechts gestapelt. Log unten volle Breite.

---

## B8. `lib/api.ts`

```typescript
const BASE = "http://127.0.0.1:8000"

export const api = {
  health:      () => fetch(`${BASE}/health`).then(r => r.json()),
  predictNext: () => fetch(`${BASE}/predict/next`).then(r => r.json()),
  reset:       () => fetch(`${BASE}/predict/reset`).then(r => r.json()),
  getStats:    () => fetch(`${BASE}/stats`).then(r => r.json()),
}
```

---

## B9. Setup Frontend

```bash
cd cockpit
npx create-next-app@latest . --typescript --tailwind --app
npx shadcn-ui@latest init
npx shadcn-ui@latest add card badge button slider progress separator

npm install recharts lucide-react
npm run dev     # läuft auf localhost:3000
```

---

# Nicht in Scope

- Datenpersistenz (kein DB, kein LocalStorage)
- Authentifizierung
- Cloud-Deployment oder Docker
- WebSocket / SSE
- Mobiles Layout
- Multi-Class-Klassifikation
- Modellvergleich (kein from-scratch Baseline)
- Echtzeit-Kameraanbindung
