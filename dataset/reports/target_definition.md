# Spoilage Target Definition — AgriChainAI Multi-Product

## Overview

Three-class classification: **FRESH (0)**, **WARNING (1)**, **SPOILED (2)**.

Target variables used for labeling are **never** included as model inputs.

---

## 1. Paneer (existing AC-CSN pipeline)

**Source files:** `dataset/20241118 DataLog.csv`, `dataset/20241230 Testing Data.xlsx`

**Primary indicator:** Total Plate Count (`tpc_log10`, log10 cfu/ml)

**Secondary indicator:** Overall Acceptability (9-point hedonic scale, when available)

**Rules** (from `spoilage_preprocessing.TARGET_CONFIG`):
- **SPOILED:** TPC ≥ 4.5 OR acceptability ≤ 6.0
- **FRESH:** TPC < 3.5 AND acceptability ≥ 7.5
- **WARNING:** otherwise
- **TPC-only fallback** when sensory scores missing (days 10–13)

**Model inputs:** daily cold-chain sensor aggregates only (17 features).

**Leakage prevention:** TPC, sensory, pH, moisture excluded from features.

---

## 2. Dried Fruit (Mendeley Tanzania dataset)

**Source files:** `dataset/dry_fruit/S 1...Physicochemical...xlsx`, `S 2...Microbial...xlsx`

**Samples modeled:** stored **dried** mango/pineapple only (`Dryer type` ≠ Fresh raw fruit).

**Primary indicator:** Total Plate Count (`tpc_cfu_g` → `tpc_log10`)

**Secondary indicator:** `storage_months` (0, 3, 6 months in source study)

**Thresholds** (project-specific, justified from dried-sample distribution):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| fresh_tpc_log10_max | 3.7 | Below month-0 dried mean (~3.74); separates low-TPC tunnel/CMD samples |
| spoiled_tpc_log10_min | 3.95 | Near 75th percentile of dried TPC; aligns with 6-month elevated loads |
| fresh_max_storage_months | 0 | Initial shelf-life timepoint in source |
| spoiled_min_storage_months | 6 | Final shelf-life timepoint showing highest mean TPC |

**Rules:**
- **SPOILED:** storage ≥ 6 months OR log10(TPC) ≥ 3.95
- **FRESH:** storage ≤ 0 months AND log10(TPC) < 3.7
- **WARNING:** intermediate storage (3 months) and/or intermediate TPC

**Model inputs (inference-time, no lab QC):**
- product subtype (Mango / Pineapple)
- storage duration (days)
- dryer type (CMD / TD)
- packaging (None / HDPE / LDPE)

**NOT in source dataset (not fabricated):** storage temperature, relative humidity.

**Reference-only (post-prediction):** moisture, water activity, pH from physicochemical table by dryer type.

**Observed class distribution (dried samples, n=24):** {'SPOILED': 12, 'WARNING': 8, 'FRESH': 4}

---

## Limitations

1. Paneer: small single-batch experiment (~70 replicates).
2. Dried fruit: only 24 dried storage cells — high variance; separate model required.
3. Dried-fruit thresholds are project-specific conversions from continuous QC data.
4. No claim of universal regulatory compliance — thresholds must be validated with domain experts.
5. Temperature/humidity unavailable for dried-fruit path; environmental risk for dried products is captured indirectly via storage duration and packaging.
