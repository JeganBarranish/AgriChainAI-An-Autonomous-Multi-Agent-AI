"""
crop_metadata.py
-----------------
[RULE-BASED / DETERMINISTIC] — no learned parameters; lookup tables and
closed-form profit arithmetic used by Stage 5 (confidence-guarded re-ranking)
and to derive PWCE class weights for Stage 3 training.

All monetary values are INR.

==========================================================================
TABLE 1: MSP_ECONOMICS — official Government of India figures
==========================================================================
Used with the Crop_recommendation.csv dataset. Every price, cost and yield
below is traceable to a primary government source; nothing is invented.

price_per_kg
    Minimum Support Price (MSP) announced by the Cabinet Committee on
    Economic Affairs, divided by 100 (MSP is quoted in Rs/quintal;
    1 quintal = 100 kg).
      - Kharif Marketing Season (KMS) 2025-26 for paddy, maize, tur,
        moong, urad, cotton  [CCEA, approved 28 May 2025]
      - Rabi Marketing Season (RMS) 2026-27 for gram  [CCEA, 1 Oct 2025]
      - Jute Marketing Season 2025-26 for raw jute TD-3  [CCEA, 23 Jan 2025]

ref_yield_t_per_ha
    CACP projected all-India yield (quintal/hectare) / 10, from
    "Annex Table 5.5: Projected Cost of Production (A2, A2+FL & C2) and
    Yield for Kharif Crops for KMS2025-26", CACP Kharif Price Policy report.
    For gram and jute, which are not in that kharif table, the all-India
    yield comes from Economic Survey "Table 1.17: Yield Per Hectare of
    Major Crops" (kg/ha / 1000). This basis difference is flagged per crop
    in the `sources` field below, because CACP cost-sample yields run
    higher than Economic Survey national averages.

cost_per_ha
    DERIVED, not quoted: A2+FL cost of production (Rs/quintal) x projected
    yield (quintal/hectare). A2+FL = all paid-out costs plus the imputed
    value of family labour. Deriving it this way keeps cost and yield on the
    same CACP sample basis, so profit = revenue - cost is internally
    consistent. It deliberately EXCLUDES C2 (which adds imputed land rent),
    so these are cash-plus-family-labour margins, not economic profit.

ref_duration_days
    Crop-duration midpoints from standard Indian agronomic extension
    literature (ICAR / TNAU Agritech crop guides). These are published
    varietal ranges, NOT per-field measurements, and are used only inside
    the rule-based Stage 5 layer where a reference table is appropriate.

Crops EXCLUDED from the 22 available in Crop_recommendation.csv: apple,
banana, blackgram is included, coconut, coffee, grapes, jute is included,
kidneybeans, lentil, mango, mothbeans, muskmelon, orange, papaya,
pomegranate, watermelon. They were dropped because no official
price + cost + yield triple could be sourced for them on a consistent
basis (most are horticultural crops outside the MSP framework, and
coconut MSP is set for milling copra rather than raw coconut). Lentil was
dropped despite having MSP and cost because CACP publishes no all-India
projected yield for it in the reports consulted.

==========================================================================
TABLE 2: LEGACY_ECONOMICS — the original 6-crop demonstration table
==========================================================================
Retained ONLY so the older crop_yield.csv / synthetic code paths keep
working. These are approximate India-representative figures for an academic
demo, NOT sourced government data, and should not be quoted as such.
"""

# --------------------------------------------------------------------------
# TABLE 1 — sourced official economics (Crop_recommendation.csv label space)
# Keys match the dataset's `label` column exactly (lowercase).
# --------------------------------------------------------------------------
MSP_ECONOMICS = {
    "rice": dict(
        ref_duration_days=135, ref_yield_t_per_ha=4.418,
        price_per_kg=23.69, cost_per_ha=69_760.0,
        sources="MSP paddy-common KMS2025-26 Rs2369/qtl; CACP yield 44.18 qtl/ha, "
                "A2+FL Rs1579/qtl; duration 120-150 d (ICAR)",
    ),
    "maize": dict(
        ref_duration_days=100, ref_yield_t_per_ha=3.878,
        price_per_kg=24.00, cost_per_ha=58_480.0,
        sources="MSP maize KMS2025-26 Rs2400/qtl; CACP yield 38.78 qtl/ha, "
                "A2+FL Rs1508/qtl; duration 90-110 d (ICAR)",
    ),
    "cotton": dict(
        ref_duration_days=165, ref_yield_t_per_ha=1.551,
        price_per_kg=77.10, cost_per_ha=79_721.0,
        sources="MSP cotton medium-staple KMS2025-26 Rs7710/qtl; CACP yield "
                "15.51 qtl/ha, A2+FL Rs5140/qtl; duration 150-180 d (ICAR)",
    ),
    "pigeonpeas": dict(
        ref_duration_days=165, ref_yield_t_per_ha=1.083,
        price_per_kg=80.00, cost_per_ha=54_562.0,
        sources="MSP tur/arhar KMS2025-26 Rs8000/qtl; CACP yield 10.83 qtl/ha, "
                "A2+FL Rs5038/qtl; duration 150-180 d (ICAR)",
    ),
    "mungbean": dict(
        ref_duration_days=68, ref_yield_t_per_ha=0.524,
        price_per_kg=87.68, cost_per_ha=30_628.0,
        sources="MSP moong KMS2025-26 Rs8768/qtl; CACP yield 5.24 qtl/ha, "
                "A2+FL Rs5845/qtl; duration 60-75 d (ICAR)",
    ),
    "blackgram": dict(
        ref_duration_days=80, ref_yield_t_per_ha=0.652,
        price_per_kg=78.00, cost_per_ha=33_343.0,
        sources="MSP urad KMS2025-26 Rs7800/qtl; CACP yield 6.52 qtl/ha, "
                "A2+FL Rs5114/qtl; duration 70-90 d (ICAR)",
    ),
    "chickpea": dict(
        ref_duration_days=105, ref_yield_t_per_ha=1.224,
        price_per_kg=58.75, cost_per_ha=45_276.0,
        # Yield basis differs: Economic Survey national average, not a CACP
        # cost-sample projection. Flagged because ES averages run lower.
        sources="MSP gram RMS2026-27 Rs5875/qtl; yield 12.24 qtl/ha from "
                "Economic Survey Tbl 1.17 (NOT CACP sample); CACP A2+FL "
                "Rs3699/qtl; duration 95-110 d (ICAR)",
    ),
    "jute": dict(
        ref_duration_days=110, ref_yield_t_per_ha=2.795,
        price_per_kg=56.50, cost_per_ha=94_667.0,
        sources="MSP raw jute TD-3 2025-26 Rs5650/qtl; yield 27.95 qtl/ha from "
                "Economic Survey Tbl 1.17 (NOT CACP sample); cost of production "
                "Rs3387/qtl (CCEA press note); duration 100-120 d (ICAR)",
    ),
}

# --------------------------------------------------------------------------
# TABLE 2 — legacy demo table (crop_yield.csv / synthetic paths only)
# NOT sourced government data. Do not cite these as official.
# --------------------------------------------------------------------------
LEGACY_ECONOMICS = {
    "Wheat":   dict(ref_duration_days=120, ref_yield_t_per_ha=3.5, price_per_kg=22, cost_per_ha=30_000),
    "Rice":    dict(ref_duration_days=130, ref_yield_t_per_ha=4.0, price_per_kg=20, cost_per_ha=35_000),
    "Maize":   dict(ref_duration_days=100, ref_yield_t_per_ha=3.2, price_per_kg=18, cost_per_ha=25_000),
    "Barley":  dict(ref_duration_days=110, ref_yield_t_per_ha=2.8, price_per_kg=19, cost_per_ha=20_000),
    "Soybean": dict(ref_duration_days=100, ref_yield_t_per_ha=1.5, price_per_kg=45, cost_per_ha=22_000),
    "Cotton":  dict(ref_duration_days=180, ref_yield_t_per_ha=1.8, price_per_kg=60, cost_per_ha=55_000),
}

# Every crop the economic layer knows about. MSP entries take precedence:
# where a crop appears in both tables the sourced official figures win.
CROP_ECONOMICS = {**LEGACY_ECONOMICS, **MSP_ECONOMICS}

# Label space of the sourced (MSP) table and of the legacy table.
MSP_CROP_LIST = list(MSP_ECONOMICS.keys())
LEGACY_CROP_LIST = list(LEGACY_ECONOMICS.keys())

# Backwards compatibility: existing crop_yield.csv / synthetic code paths and
# the diagnostic script import CROP_LIST / CROP_METADATA expecting the 6-crop
# legacy space.
CROP_LIST = LEGACY_CROP_LIST
CROP_METADATA = {
    c: (v["ref_duration_days"], v["ref_yield_t_per_ha"], v["price_per_kg"], v["cost_per_ha"])
    for c, v in LEGACY_ECONOMICS.items()
}


def _entry(crop: str) -> dict:
    if crop not in CROP_ECONOMICS:
        raise ValueError(
            f"Unknown crop '{crop}'. Known crops: {sorted(CROP_ECONOMICS)}")
    return CROP_ECONOMICS[crop]


def get_expected_profit(crop: str, predicted_yield_t_per_ha: float,
                        land_area_ha: float = 1.0) -> float:
    """
    Rule-based economic layer (Stage 5).
    Profit = (yield * price) - cultivation cost, scaled by land area.
    Falls back to the reference yield when no positive prediction is given.
    """
    e = _entry(crop)
    yield_t_per_ha = (predicted_yield_t_per_ha if predicted_yield_t_per_ha > 0
                      else e["ref_yield_t_per_ha"])
    revenue = yield_t_per_ha * 1000 * e["price_per_kg"]   # tonnes -> kg
    return round((revenue - e["cost_per_ha"]) * land_area_ha, 2)


def get_reference_duration(crop: str) -> int:
    return _entry(crop)["ref_duration_days"]


def get_reference_yield(crop: str) -> float:
    """
    Reference (table) yield in t/ha. Stage 5 uses this wherever the model did
    not itself predict a yield, so a table estimate is never presented as a
    model prediction.
    """
    return _entry(crop)["ref_yield_t_per_ha"]


def get_source_note(crop: str) -> str:
    """Provenance string for a crop's economics ('' for legacy demo values)."""
    return _entry(crop).get("sources", "legacy demonstration value (not sourced)")


def get_reference_profit(crop: str) -> float:
    """Reference profit per hectare at the table yield."""
    return get_expected_profit(crop, 0.0, land_area_ha=1.0)


def get_profit_weight_vector(crop_list=None, clip_range=(0.5, 2.0)):
    """
    Per-class weights, in `crop_list` order, for the Profit-Weighted
    Cross-Entropy (PWCE) loss on the crop head.

    Weight for class c = reference profit-per-hectare of c, normalized to
    mean 1.0 so the weighted loss stays on a comparable scale to plain
    cross-entropy instead of being dominated by whichever crop has the
    largest absolute profit.

    clip_range caps the spread AFTER normalization and BEFORE renormalizing.
    Rationale: PWCE is meant to break ties between agronomically plausible
    crops, not to override agronomy. On the sourced MSP table the raw spread
    reaches about 4.1x (jute vs mungbean, because jute's per-hectare margin
    is roughly four times mungbean's), which is enough to bias the argmax on
    genuinely ambiguous samples. Capping to [0.5, 2.0] preserves the
    profit ORDERING while bounding its influence. Pass clip_range=None to
    disable and report the uncapped behaviour.
    """
    crop_list = crop_list if crop_list is not None else CROP_LIST
    profits = [max(get_reference_profit(c), 1.0) for c in crop_list]

    mean_profit = sum(profits) / len(profits)
    weights = [p / mean_profit for p in profits]

    if clip_range is not None:
        lo, hi = clip_range
        weights = [min(max(w, lo), hi) for w in weights]
        mean_w = sum(weights) / len(weights)
        weights = [w / mean_w for w in weights]   # restore mean 1.0

    return weights
