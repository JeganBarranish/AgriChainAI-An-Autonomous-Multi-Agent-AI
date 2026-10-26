"""
Routing / state-machine tests use lightweight TEST DOUBLES for the services so
they run instantly. They never touch the app: the real-model smoke tests live in
tests/test_integration_real_models.py.
"""

import pytest

from agents.fields import Field
from agents.orchestrator import ChatState, Orchestrator, Task, detect_intent


class Stub:
    def __init__(self, key, fields, status="Ready", error=None, fail=None):
        self.key, self._fields, self.status, self.error, self.fail = key, fields, status, error, fail
        self.api_name = f"stub.{key}()"
        self.agent_name = "Farm Agent" if key == "farm" else "Spoilage Agent"
        self.calls = []

    def fields(self, collected=None):
        return list(self._fields)

    def _go(self, collected):
        self.calls.append(dict(collected))
        if self.fail:
            raise self.fail
        return {"inputs": dict(collected), "stub": self.key}

    def recommend(self, collected):
        out = self._go(collected)
        out.update(best={"crop": "x", "confidence": .5, "expected_yield_t_per_ha": 1, "expected_profit_inr": 1},
                   alternatives=[], dataset="stub", caveats=[])
        return out

    def predict(self, collected):
        out = self._go(collected)
        out.update(product_subtype=self.key, status="FRESH", spoilage_risk=1.0, probabilities={}, notes=[],
                   interpretation="", model_used="stub")
        return out


def make(**kw):
    farm = Stub("farm", [Field("N", "Nitrogen", "float", "N?", minimum=0, maximum=300),
                         Field("sowing", "Sowing", "date", "date?", optional=True, skip_note="today")],
                **kw.get("farm", {}))
    paneer = Stub("paneer", [Field("temperature", "Temperature", "float", "T?", minimum=-30, maximum=40),
                             Field("fridge", "Fridge", "int", "F?", minimum=0, maximum=100, optional=True,
                                   skip_note="median")], **kw.get("paneer", {}))
    dried = Stub("dried", [Field("product_subtype", "Fruit", "choice", "fruit?", choices=("Mango", "Pineapple")),
                           Field("days", "Days", "float", "days?", minimum=0, maximum=730)], **kw.get("dried", {}))
    o = Orchestrator(farm, paneer, dried)
    s = ChatState()
    o.new_conversation(s)
    return o, s, farm, paneer, dried


def last(s):
    return s.messages[-1]


def texts(msg):
    return " ".join(b.get("text", "") for b in msg["blocks"])


def types(msg):
    return [b["type"] for b in msg["blocks"]]


@pytest.mark.parametrize("text,task", [
    ("Suggest a profitable crop.", Task.FARM_RECOMMENDATION),
    ("I want a crop recommendation.", Task.FARM_RECOMMENDATION),
    ("Check whether my paneer is spoiled.", Task.PANEER_SPOILAGE),
    ("Predict spoilage for my dried mango.", Task.DRIED_FRUIT_SPOILAGE),
    ("Assess a batch of dried pineapple.", Task.DRIED_FRUIT_SPOILAGE),
    ("Run a farm recommendation and spoilage check.", Task.COMBINED_ANALYSIS),
    ("Run both analyses", Task.COMBINED_ANALYSIS),
])
def test_intent_variants(text, task):
    assert detect_intent(text).task == task


def test_ambiguous_request_asks_clarifying_question():
    o, s, *_ = make()
    o.handle_message(s, "recommend")
    assert s.workflow is None and s.awaiting == {"kind": "intent"}
    assert "not sure" in texts(last(s)).lower()


def test_unsupported_request_lists_capabilities():
    o, s, *_ = make()
    o.handle_message(s, "what's the weather tomorrow")
    assert s.current_task == Task.UNSUPPORTED_REQUEST and "currently help" in texts(last(s))


def test_farm_flow_validation_skip_and_call():
    o, s, farm, *_ = make()
    o.handle_message(s, "Recommend a crop")
    assert s.workflow.key == "farm"
    o.handle_message(s, "lots")                        # invalid -> re-prompt, no call
    assert "error" in types(last(s)) and farm.calls == []
    o.handle_message(s, "90")
    o.handle_message(s, "skip")                        # optional
    assert farm.calls == [{"N": 90.0, "sowing": None}]
    assert "farm_result" in types(last(s)) and s.workflow is None
    steps = [t["step"] for t in last(s)["trace"]]
    assert "stub.farm() executed" in steps and last(s)["trace"][-2]["status"] == "SUCCESS"


def test_dried_fruit_prefill_from_request():
    o, s, _, _, dried = make()
    o.handle_message(s, "Predict spoilage for my dried mango")
    assert s.workflow.collected["product_subtype"] == "Mango" and s.workflow.idx == 1
    o.handle_message(s, "30")
    assert dried.calls == [{"product_subtype": "Mango", "days": 30.0}]


def test_spoilage_without_product_asks_which_product():
    o, s, *_ = make()
    o.handle_message(s, "check spoilage")
    assert s.awaiting == {"kind": "product", "combined": False}
    o.handle_message(s, "paneer")
    assert s.workflow.key == "paneer"


def test_cancel_discards_inputs():
    o, s, farm, *_ = make()
    o.handle_message(s, "Recommend a crop")
    o.handle_message(s, "90")
    o.handle_message(s, "cancel")
    assert s.workflow is None and farm.calls == []
    o.handle_message(s, "Recommend a crop")             # nothing leaks from the cancelled run
    assert s.workflow.collected == {} and s.workflow.idx == 0


def test_switch_task_mid_workflow_no_leak():
    o, s, farm, paneer, _ = make()
    o.handle_message(s, "Recommend a crop")
    o.handle_message(s, "90")
    o.handle_message(s, "actually check my paneer")
    assert s.workflow.key == "paneer" and s.workflow.collected == {}
    o.handle_message(s, "4")
    o.handle_message(s, "skip")
    assert paneer.calls == [{"temperature": 4.0, "fridge": None}] and farm.calls == []


def test_new_conversation_clears_everything():
    o, s, *_ = make()
    o.handle_message(s, "Recommend a crop")
    o.handle_message(s, "90")
    o.new_conversation(s)
    assert s.workflow is None and s.awaiting is None and s.results == {} and len(s.messages) == 1


def test_repeat_analysis_without_stale_inputs():
    o, s, farm, *_ = make()
    for n in ("10", "20"):
        o.handle_message(s, "Recommend a crop")
        o.handle_message(s, n)
        o.handle_message(s, "skip")
    assert [c["N"] for c in farm.calls] == [10.0, 20.0]


def test_combined_flow_both_agents_and_summary():
    o, s, farm, paneer, _ = make()
    o.handle_message(s, "Run both analyses")
    o.handle_message(s, "90")
    o.handle_message(s, "skip")
    assert "farm_result" in types(last(s)) and s.awaiting == {"kind": "product", "combined": True}
    o.handle_message(s, "Paneer")
    o.handle_message(s, "4")
    o.handle_message(s, "skip")
    t = types(last(s))
    assert "spoilage_result" in t and "summary" in t
    assert farm.calls and paneer.calls


def test_combined_with_product_named_upfront():
    o, s, *_, dried = make()
    o.handle_message(s, "farm recommendation and dried pineapple spoilage")
    o.handle_message(s, "90")
    o.handle_message(s, "skip")
    assert s.workflow.key == "dried" and s.workflow.collected["product_subtype"] == "Pineapple"


def test_model_exception_is_reported_not_raised():
    o, s, *_ = make(paneer={"fail": RuntimeError("boom")})
    o.handle_message(s, "check paneer")
    o.handle_message(s, "4")
    o.handle_message(s, "skip")
    assert "error" in types(last(s)) and s.workflow is None
    assert last(s)["trace"][-2]["status"] == "FAILED"


def test_unavailable_agent_is_reported():
    o, s, *_ = make(dried={"status": "Not configured", "error": "Missing artifact"})
    o.handle_message(s, "check dried mango")
    assert s.workflow is None and "not available" in texts(last(s))
    assert o.statuses()["Dried-Fruit Spoilage"][0] == "Not configured"


def test_farm_error_blocks_farm_task():
    o, s, *_ = make(farm={"status": "Error", "error": "dataset missing"})
    o.handle_message(s, "recommend a crop")
    assert s.workflow is None and "dataset missing" in texts(last(s))
