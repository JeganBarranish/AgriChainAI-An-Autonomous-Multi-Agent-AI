# AgriChainAI — First Review

Two ML systems built on real experimental data:

1. **Paneer Cold-Chain Spoilage Prediction** (AC-CSN + baselines)
2. **AI4I Predictive-Maintenance Anomaly Detection** (Random Forest / XGBoost / LightGBM)

---

## Paneer Cold-Chain Spoilage Prediction

### 1. Why the Paneer dataset

The system predicts dairy (paneer) deterioration from cold-chain storage
conditions. It replaces an earlier prototype that used unrelated data. The
new model is trained exclusively on **real experimental measurements** from a
paneer refrigerated-storage study — no synthetic data.

### 2. Cold-chain sensor data (`dataset/20241118 DataLog.csv`)

1,950 sensor readings logged roughly every 10 minutes from 4–18 Nov 2024
(15 calendar days):

| Column | Meaning |
|---|---|
| `Time_Stamp` | Reading time (`DD-MM-YYYY HH:MM`) |
| `Temperature` | Fridge temperature (°C) |
| `RH` | Relative humidity (%) |
| `Compressor` | Compressor state (0/1) |
| `Fridge_Open` | Door-open flag (0/1) |

### 3. Paneer quality data (`dataset/20241230 Testing Data.xlsx`)

Daily product-quality observations for storage days 0–13, five replicates per
day (70 replicate rows):

- **Microbiological** (log10 cfu/ml): Total Plate Count (TPC), coliform,
  yeast & mold, psychrophilic count
- **Sensory** (9-point hedonic, days 0–9 only): colour/appearance,
  body/texture, flavour, overall acceptability
- **Physicochemical**: pH, moisture %

### 4. Dataset synchronization

Sensor readings are **aggregated per storage day** (not treated as
independent samples) and joined to quality replicates on `storage_day`.
Daily cold-chain features:

- Temperature: mean, min, max, std, range, excursion count, time above 4 °C
- Humidity: mean, min, max, std, range
- Compressor: on-ratio, cycle count (0→1 transitions)
- Fridge: open count, open ratio
- Time: storage day

Result: `dataset/Paneer_ColdChain_Supervised.csv` — 70 samples
(14 days × 5 replicates). Originals are never modified.

### 5. Target class definition

Three classes derived from real quality evidence (full methodology in
`results/target_definition.json`; thresholds configurable in
`spoilage_preprocessing.py`):

| Class | Rule |
|---|---|
| **SPOILED (2)** | TPC ≥ 4.5 log10 cfu/ml OR overall acceptability ≤ 6.0 |
| **FRESH (0)** | TPC < 3.5 log10 cfu/ml AND acceptability ≥ 7.5 |
| **WARNING (1)** | everything in between |

TPC is the primary microbiological spoilage indicator; sensory acceptability
is the secondary confirmation. For days 10–13 (no per-replicate sensory data)
TPC-only rules apply. Observed distribution: FRESH 25, WARNING 15, SPOILED 30.

### 6. Data-leakage prevention

- TPC, coliform, yeast/mold, psychrophilic counts, sensory scores, pH and
  moisture are used **only** for target construction — never as model inputs.
- Model inputs are limited to cold-chain sensor aggregates + storage duration
  (information realistically available during storage/transport).
- Train/test split is **grouped by storage day**: all 5 replicates of a day
  stay on the same side of the split (test days: 2, 6, 9, 12 → 20 samples,
  covering all three classes).

### 7. AC-CSN architecture

**AgriChain Cold-Chain Spoilage Network** — a small dual-branch neural
network (~1.7k parameters) sized for the dataset:

```
sequence (last 3 days × 17 features) ──► GRU(12) ──► Dropout(0.3) ─┐
environmental features (12) ──► Dense(8, relu) ────────────────────┤──► Concat
operational features (5) ──► Dense(6, relu) ───────────────────────┘     │
                                                     Dense(16, relu) ◄───┘
                                                     Dropout(0.3)
                                                     Dense(3, softmax)
```

BatchNormalization was removed after experiments showed batch statistics were
too noisy with ~50 training samples.

### 8. Baseline models

Random Forest, XGBoost, LightGBM (flat daily features) and a plain GRU
(same 3-day sequences), all trained on the identical dataset and split.

### 9. Evaluation methodology

Identical held-out test set (days 2, 6, 9, 12; 20 samples). Metrics:
accuracy, macro precision/recall/F1, weighted F1, and per-class recall.
Model selection uses **Macro F1 first, Spoiled-class recall as tiebreaker** —
not accuracy alone.

### 10. Results (held-out test set)

| Model | Accuracy | Macro-F1 | Spoiled Recall |
|---|---|---|---|
| Random Forest | 0.750 | 0.778 | 0.500 |
| **XGBoost (selected)** | **1.000** | **1.000** | **1.000** |
| LightGBM | 1.000 | 1.000 | 1.000 |
| GRU baseline | 0.750 | 0.778 | 0.500 |
| AC-CSN | 0.750 | 0.778 | 0.500 |

**XGBoost was selected as the final model** — it genuinely outperformed
AC-CSN on the unseen test days. AC-CSN is also saved and reloadable for
future work on larger datasets.

### 11. Limitations

- Only 70 supervised samples from a **single storage experiment**; perfect
  test scores reflect the strong monotonic day–spoilage relationship in this
  study and **do not imply generalization** to other batches, products, or
  storage regimes.
- Sensory data missing for days 10–13 (TPC-only labels there).
- Class thresholds are data-driven and configurable; validate with domain
  experts before production use.

### 12. Example prediction output

```
┌─────────────────────────────────────┐
│         COLD CHAIN ANALYSIS         │
├─────────────────────────────────────┤
│ Temperature:          3.9 °C        │
│ Humidity:            36.9 %         │
│ Storage duration:     8 days        │
├─────────────────────────────────────┤
│ Spoilage Risk:       98.3 %         │
│ Status:            SPOILED          │
│                                     │
│ Class probabilities:                │
│   Fresh       1.2 %                 │
│   Warning     1.0 %                 │
│   Spoiled    97.9 %                 │
└─────────────────────────────────────┘
```

Spoilage risk formula: `risk % = 0·P(FRESH) + 50·P(WARNING) + 100·P(SPOILED)`.

### Files

| File | Purpose |
|---|---|
| `spoilage_preprocessing.py` | Sensor aggregation, quality parsing, target labels |
| `spoilage_training.py` | Trains AC-CSN + 4 baselines, compares, saves best |
| `spoilage_predict.py` | Inference CLI + reusable `SpoilagePredictor` class |
| `models/ac_csn_spoilage_model.keras` | Trained AC-CSN network |
| `models/best_spoilage_model.pkl` | Selected best model (XGBoost) |
| `models/spoilage_preprocessor.pkl` | Scaler + feature/label metadata |
| `results/target_definition.json` | Target methodology & thresholds |
| `results/spoilage_*.png/json` | Evaluation charts and metrics |

### Run

```bash
pip3 install -r requirements.txt
python3 spoilage_preprocessing.py   # build supervised dataset
python3 spoilage_training.py        # train + compare all models
python3 spoilage_predict.py         # run a real inference
```

---

## AI4I Anomaly Detection (unchanged)

`anomaly_detection.py` trains Random Forest, XGBoost, and LightGBM on
`dataset/Preprocessed_Predictive_Maintenance.csv` to detect machine failures.
Best model (LightGBM, F1 = 0.75) is saved as `anomaly_model.pkl`; evaluation
charts are in `results/`.

```bash
python3 anomaly_detection.py
```
