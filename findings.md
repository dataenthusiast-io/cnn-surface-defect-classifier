# Findings — Was funktioniert, was nicht, was ungenutzt ist

Zusammenfassung aller Analyse-Ergebnisse (Stand: Juni 2026). Details zur ML-Methodik in [`analyse.md`](analyse.md).

---

## 1. Was aktuell funktioniert ✅

| Komponente | Status | Anmerkung |
|---|---|---|
| **Daten-Staging** | Funktioniert | `data/raw/` (1783 JPEGs in 4 Klassen) + `data/annotations/` (1783 XMLs) entpackt und vollständig; passt exakt zum Layout aus `DATA_REQUIREMENTS.md` |
| **Training** (`src/train.py`) | Funktioniert | 2-Phasen Transfer Learning lief sauber durch: 23 Epochen, Early Stop, Best Val F1 = 0,945 (Epoche 18). Resume-Checkpointing, CSV-Logging, gewichtete Loss — alles aktiv |
| **Evaluation** (`src/evaluate.py`) | Funktioniert | 96,3 % Test-Accuracy, 10/270 Fehler; alle 5 Plots wurden erzeugt |
| **Inference-API** (FastAPI, Port 8000) | Funktioniert | `/health` liefert `{"status":"ok", "model":"best_model.pth", "device":"mps", "test_size":270}`; CORS für Port 3000 konfiguriert |
| **Cockpit** (Next.js) | Funktioniert | Verbindet sich korrekt mit der API — sofern es auf Port 3000 läuft (siehe Befund 2.1) |
| **Grad-CAM** (`api/pipeline.py`) | Funktioniert im API-Pfad | Algorithmisch korrekt (Hooks auf layer4, ReLU-gewichtete Aktivierungen). Heatmaps fokussieren plausibel aufs Bauteil statt auf den Hintergrund. Getestet mit je 1 Sample pro Klasse |
| **Branch-Struktur** | Sauber | Lineare Kette: `main` → `feature/adapt-production-data-4-class` → `feature/gradcam-explainability`. Merge-Reihenfolge: erst adapt, dann gradcam |

## 2. Was nicht (oder nur zufällig) funktioniert ⚠️

### 2.1 Behoben während der Session
- **„Modell offline" im Cockpit**: Ursache war ein Port-Konflikt — eine fremde Next.js-App belegte Port 3000, das Cockpit wich auf 3001 aus, und die API erlaubt per CORS nur Origin 3000. Der Browser blockierte den `/health`-Fetch → UI zeigte „Offline". **Fix:** fremden Prozess beendet; Cockpit muss auf Port 3000 neu gestartet werden.
- **`Daten.zip` ließ sich nicht entpacken**: macOS-`unzip` und `bsdtar` scheitern an Zip64-Archiven > 4 GB (Archiv war intakt). **Fix:** mit Pythons `zipfile` entpackt.

### 2.2 Offene Probleme im Code

| # | Problem | Ort | Schwere |
|---|---|---|---|
| 1 | **Daten-Leakage durch Nahezu-Duplikate**: Empirisch belegt (dHash-Replikation des Splits, Seed 42): 25/25 Abdruck-1- und 23/24 Stanzfehler-Testbilder haben ein Nahezu-Duplikat im Trainingsset (teils Distanz 0). Die 96 % Accuracy messen Wiedererkennung, nicht Generalisierung | `dataset.py:91` (Split auf Bild- statt Teil-Ebene) | **Kritisch** |
| 2 | **Latenter Crash-Bug in Grad-CAM**: fehlendes `.detach()` → `RuntimeError` bei jedem Aufruf außerhalb von `@torch.no_grad()`. Funktioniert in der API nur, weil `next()` den no_grad-Kontext liefert | `pipeline.py:89` | Hoch |
| 3 | **Split wird nicht persistiert**: API rekonstruiert den Test-Split zur Laufzeit aus Ordnerinhalt + Seed. Ein einziges neues Bild verschiebt den Split stillschweigend → API „testet" auf Trainingsbildern | `dataset.py:158` (`get_test_dataset`) | Hoch |
| 4 | **Grad-CAM-Region bei i.O.-Teilen irreführend**: „Kein Defekt" hat keinen Ort — die Region umfasst dann fast das ganze Bild (`x:0, width:1.0`), das Cockpit zeigt sie trotzdem als Defektregion an | `pipeline.py:104–113` + UI | Mittel |
| 5 | **Per-Bild-Normalisierung der Heatmap** (`cam / cam.max()`): erzeugt immer einen knallroten Hotspot, auch bei schwacher absoluter Aktivierung — Intensität und Modellunsicherheit sind entkoppelt | `pipeline.py:90` | Mittel |
| 6 | **Eine Box für alle Aktivierungen**: getrennte Aktivierungszonen verschmelzen zu einer Riesenbox (beobachtet: 79 % Bildbreite) | `pipeline.py:104–113` | Mittel |
| 7 | **Unkalibrierte Konfidenz**: Softmax klebt bei 1,0; beobachtet: Fehlklassifikation Stanzfehler→Abdruck 1 mit 93,5 % Konfidenz. Cockpit zeigt diese Werte Bedienern an | Modell/`pipeline.py` | Mittel |
| 8 | **Unvollständige Reproduzierbarkeit**: Seed 42 gilt nur für den Split; Torch-Init, DataLoader-Shuffle und Augmentierung sind ungeseedet | `config.py:19`, `train.py` | Niedrig |
| 9 | **`CLASS_NAMES` doppelt gepflegt** (model-training + inference-api), manuelle Synchronisation nötig; Kommentar „Python-alphabetical" in `config.py:21` irreführend | beide `config.py` | Niedrig |
| 10 | **Anisotropes `Resize((224,224))`**: 4:3-Bilder werden gestaucht, runde Teile werden Ellipsen | `dataset.py:23` | Niedrig |

## 3. Was vorhanden ist, aber nicht genutzt wird 📦

| Artefakt | Umfang | Befund |
|---|---|---|
| **`data/annotations/` (Pascal-VOC-XMLs)** | 1783 Dateien, 7 MB | **Wird nirgends gelesen** — im gesamten Code-Tree existiert kein einziger Lesezugriff (nur das Erzeuger-Skript `rename_and_annotate.py` schreibt sie). Zusätzlich inhaltsleer: jede Bounding Box umspannt das Vollbild (0,0–4032,3024) statt der Defektregion. Für die Klassifikation nicht nötig (Label = Ordnername); für die angedachte Object-Detection-Erweiterung müssten sie komplett neu annotiert werden. Ironie: echte Defekt-Boxen wären genau die Ground Truth, die zur Validierung der Grad-CAM-Heatmaps fehlt |
| **Test-Loader im Training** | — | `train.py:182` erzeugt ihn und verwirft ihn bewusst (`_`) — das ist methodisch **richtig** (Test-Set bleibt unangetastet), nur der Vollständigkeit halber gelistet |
| **`NUM_WORKERS = 0`** | `config.py:15` | Verschenktes Potenzial: jedes JPEG wird in jeder Epoche im Hauptprozess neu dekodiert (`dataset.py:79`) — ein Großteil der ~78 s/Epoche. Mehr Worker würden genau diesen Schritt parallelisieren |
| **`predict.py`** | `src/predict.py` | Einzelbild-Inferenz-Skript, wird von keinem anderen Modul importiert (Standalone-Utility) |
| **Inference-API als „Service"** | `inference-api/` | Kein Upload-Endpoint — die API iteriert nur über das lokale Test-Set. Es ist eine Demo-Simulation, kein echter Inferenz-Service |

## 4. Priorisierte nächste Schritte

1. **Ehrliches Test-Set**: separate Aufnahmesession (andere Teile, anderer Tag/Hintergrund) → belastbare Generalisierungszahl; der zu erwartende Einbruch ist die spannendste Erkenntnis für den Bericht
2. **`.detach()`-Fix** in `pipeline.py:89` (Einzeiler)
3. **Split-Manifest** beim Training speichern (JSON mit Dateinamen) und in der API laden
4. Grad-CAM-Region bei i.O.-Vorhersagen im Cockpit ausblenden
5. Bei künftiger Datenerhebung **Teil-IDs erfassen** → Group-Split möglich
6. Konfidenz kalibrieren (Temperature Scaling), Seeds vervollständigen, `NUM_WORKERS` erhöhen
