"""Smoke tests against the REAL saved models / trained FarmAgent (no stubs)."""

import warnings

import pytest

from agents.farm_adapter import FarmService
from agents.orchestrator import ChatState, Orchestrator
from agents.spoilage_adapter import DryFruitService, PaneerService

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def farm():
    svc = FarmService.initialize()          # trains the existing FarmAgent once for the module
    if svc.status != "Ready":
        pytest.skip(f"Farm Agent unavailable: {svc.error}")
    return svc


@pytest.fixture(scope="module")
def paneer():
    svc = PaneerService()
    if svc.status != "Ready":
        pytest.skip(svc.error)
    return svc


@pytest.fixture(scope="module")
def dried():
    svc = DryFruitService()
    if svc.status != "Ready":
        pytest.skip(svc.error)
    return svc


def run(o, s, *msgs):
    for m in msgs:
        o.handle_message(s, m)
    return s.messages[-1]


def blocks(msg, t):
    return [b for b in msg["blocks"] if b["type"] == t]


def test_farm_schema_matches_active_spec(farm):
    keys = [f.key for f in farm.fields()]
    assert keys == ["N", "P", "K", "ph", "temperature", "humidity", "rainfall", "land_area_ha", "sowing_date"]


def test_farm_recommendation_matches_direct_call(farm, paneer, dried):
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "Recommend a crop", "90", "42", "43", "6.5", "21", "82", "200", "2", "skip")
    d = blocks(msg, "farm_result")[0]["data"]
    direct = farm.agent.recommend(N=90.0, P=42.0, K=43.0, ph=6.5, temperature=21.0, humidity=82.0,
                                  rainfall=200.0, land_area_ha=2.0)
    assert d["best"]["crop"] == direct["best_crop"]["crop"]
    assert d["best"]["expected_profit_inr"] == direct["best_crop"]["expected_profit_inr"]
    assert len(d["alternatives"]) == 3
    assert any(st["step"] == "FarmAgent.recommend() executed" and st["status"] == "SUCCESS" for st in msg["trace"])


def test_farm_rejects_invalid_input(farm, paneer, dried):
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "Recommend a crop", "-5")
    assert blocks(msg, "error") and s.workflow.idx == 0


def test_paneer_flow_matches_predictor(farm, paneer, dried):
    from spoilage.predictor import predict_paneer
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "Check whether my paneer is spoiled.", "8", "4.2", "37", "skip", "skip")
    d = blocks(msg, "spoilage_result")[0]["data"]
    direct = predict_paneer(temperature=4.2, humidity=37.0, storage_duration=8.0)
    assert d["probabilities"]["SPOILED"] == pytest.approx(direct["spoiled_probability"])
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert d["status"] == direct["status"] and d["spoilage_risk"] == direct["spoilage_risk"]


def test_dried_fruit_options_come_from_artifact_and_match_predictor(farm, paneer, dried):
    from spoilage.predictor import predict_dryfruit
    assert dried.valid["fruit"] == ["Mango", "Pineapple"]
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "Predict spoilage for my dried pineapple", "90", "TD", "LDPE")
    d = blocks(msg, "spoilage_result")[0]["data"]
    direct = predict_dryfruit("Pineapple", 90.0, "TD", "LDPE")
    assert d["probabilities"]["WARNING"] == pytest.approx(direct["warning_probability"])
    assert d["product_subtype"] == "Dried Pineapple"


def test_unsupported_fruit_rejected(farm, paneer, dried):
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "check dried fruit spoilage", "Banana")
    assert blocks(msg, "error")


def test_combined_real_workflow(farm, paneer, dried):
    o, s = Orchestrator(farm, paneer, dried), ChatState()
    o.new_conversation(s)
    msg = run(o, s, "Run both analyses", "80", "45", "20", "6.8", "24", "65", "90", "1", "skip",
              "dried mango", "60", "CMD", "HDPE")
    assert blocks(msg, "spoilage_result") and blocks(msg, "summary")
    assert "farm" in s.results and "spoilage" in s.results
    agents = {st["agent"] for st in msg["trace"]}
    assert "Spoilage Agent" in agents
