# Third-review integration audit

Source audited: public repo clone at commit `8a11daf` (second-review merge). The six
files uploaded from the local workspace (`farm_agent.py`, `hybrid_model.py`,
`crop_metadata.py`, `dataset.py`, `spoilage_preprocessing.py`, `spoilage_training.py`)
and `requirements.txt` are byte-identical to that commit. Anything in the local
workspace that is not in the public repo (extra models, edited README) was NOT visible.

## A. Farm Agent (`farm_agent.py`)
- `FarmAgent(seed=42, spec=None)`; `train(epochs=150, data_source="crop_reco", max_samples, search=True, verbose=True, cv_folds=5)`
  re-initialises the agent with the dataset's spec. **No save/load API exists**, so the model is
  trained once per Streamlit process (`st.cache_resource`) using the agent's own defaults
  (search on; only the CV *report* is skipped, which does not change the deployed model).
- Active schema (`crop_reco`): `N, P, K, ph, temperature, humidity, rainfall`, plus `land_area_ha`
  and optional `sowing_date`. The UI form is built from `agent.spec`; a legacy-spec form
  (region, soil_type, month, irrigation, fertilizer, optional water) is also generated if the
  agent were trained on that spec (code path implemented, not exercised in tests).
- `recommend(**features)` returns `{best_crop{crop, confidence, expected_yield_t_per_ha,
  expected_profit_inr, estimate_source, harvest_month, harvest_date_estimate}, alternatives[3]}`.
- Label space: rice, maize, cotton, pigeonpeas, mungbean, blackgram, chickpea, jute. **Mango/
  pineapple are not recommendable crops**, so the combined workflow does not link the two agents.
- This dataset has no yield column: yield/profit come from the MSP/CACP reference table
  (`estimate_source = reference_table_estimate`). The UI states this.

## B. Spoilage API (`spoilage/predictor.py`, already callable — no refactor needed)
- `predict_paneer(temperature, humidity, storage_duration, compressor_activity=None, fridge_opening_count=None)`
- `predict_dryfruit(product_subtype, storage_duration_days, dryer_type, packaging)`
- Output keys: `fresh/warning/spoiled_probability`, `status`, `spoilage_risk`, `confidence`, `model_used`, inputs.
- `spoilage/inference.py` is the CLI over the same functions and is untouched. `spoilage_predict.py`
  is the older first-review script and is not used.

## C. Dried fruit
Supported: **Mango, Pineapple**; dryer **CMD, TD**; packaging **None, HDPE, LDPE** (read from
`models/dryfruit_preprocessor.pkl` at runtime, not hard-coded). No temperature/humidity inputs.

## D. Artifacts (all load without retraining)
`models/paneer_preprocessor.pkl`, `paneer_best_model.pkl` (XGBoost), `dryfruit_preprocessor.pkl`,
`dryfruit_best_model.pkl` (XGBoost). The `.keras` AC-CSN files are only used when no XGBoost
pickle exists, so TensorFlow is not needed for the current artifacts (I ran without it).
Farm Agent: nothing persisted → trained at startup.

## E. Existing UI/orchestration
None found. Added `agents/` + `app.py`.

## Findings the review panel may ask about (not changed — reported)
1. `predict_paneer` computes `compute_spoilage_risk` (0/50/100 weighted) but returns
   `spoilage_risk = P(SPOILED) × 100`. The UI labels it exactly that way.
2. Paneer training data cover only mean temperature 3.74–4.43 °C, humidity 36.7–37.4 %, days 0–13.
   Observed: temperature 6 °C on day 3 → FRESH (1.4 % spoiled) but day 8 → SPOILED (96.6 %): the
   prediction is driven mostly by storage duration. Out-of-range inputs raise an on-screen warning.
3. Dried-fruit model: 24 rows (12 per fruit), storage 0–180 days. Reported metrics of 1.0 come
   from a handful of test rows and should not be presented as general accuracy.
4. Scalers were pickled with scikit-learn 1.6.1 and XGBoost models with an older XGBoost;
   they load and run (version warnings suppressed in the UI). Pin versions for a clean demo machine.
5. Paneer `inputs_used` marks a value as "default" if it equals the training median exactly.
