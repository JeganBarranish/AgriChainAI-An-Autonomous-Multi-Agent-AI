"""
Supervisor / orchestrator for AgriChainAI.

Deterministic (keyword + conversation-state) routing -- NO LLM and no network.
The orchestrator never predicts anything itself: it routes to the Farm Agent or
the Spoilage Agent services, validates inputs, calls their real prediction
functions, records an honest execution trace, and returns display blocks.

UI-independent: it only manipulates a ChatState, so it is testable without
Streamlit.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from agents.fields import Field, ValidationError, is_skip, parse_value


class Task(str, Enum):
    FARM_RECOMMENDATION = "FARM_RECOMMENDATION"
    PANEER_SPOILAGE = "PANEER_SPOILAGE"
    DRIED_FRUIT_SPOILAGE = "DRIED_FRUIT_SPOILAGE"
    COMBINED_ANALYSIS = "COMBINED_ANALYSIS"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"


# ------------------------------------------------------------------ intent
_R = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
_CANCEL = _R(r"\b(cancel|abort|stop|never ?mind|quit)\b")
_RESET = _R(r"\b(new (chat|conversation)|start (over|again|afresh)|reset)\b")
_HELP = _R(r"\b(help|what can you do|capabilit\w+|hello|hi|hey|options)\b")
_FARM = _R(r"\b(crops?|farm\w*|sow\w*|plant\w*|harvest\w*|cultivat\w*|agronom\w*)\b")
_PANEER = _R(r"\b(paneer|cottage cheese)\b")
_DRIED = _R(r"\b(dried|dry[- ]?fruits?|dehydrated|dried[- ]?fruits?)\b")
_FRUIT = _R(r"\b(mango(es)?|pineapples?)\b")
_SPOIL = _R(r"\b(spoil\w*|shelf[- ]?life|cold[- ]?chain|rott\w*|storage|stored|fresh\w*)\b")
_BOTH = _R(r"\b(both|all analyses|everything)\b")
_VAGUE = _R(r"\b(recommend\w*|suggest\w*|check|predict\w*|assess\w*|analy[sz]\w*|run|test)\b")


@dataclass
class Intent:
    farm: bool = False
    paneer: bool = False
    dried: bool = False
    fruit: str | None = None
    spoil: bool = False
    both: bool = False
    cancel: bool = False
    reset: bool = False
    help: bool = False
    vague: bool = False

    @property
    def task(self) -> Task | None:
        if self.both or (self.farm and (self.paneer or self.dried or self.spoil)):
            return Task.COMBINED_ANALYSIS
        if self.farm:
            return Task.FARM_RECOMMENDATION
        if self.paneer:
            return Task.PANEER_SPOILAGE
        if self.dried:
            return Task.DRIED_FRUIT_SPOILAGE
        return None

    @property
    def product(self) -> str | None:
        return "paneer" if self.paneer else ("dried" if self.dried else None)


def detect_intent(text: str) -> Intent:
    t = text or ""
    fruit_m = _FRUIT.search(t)
    farm = bool(_FARM.search(t))
    i = Intent(
        farm=farm,
        paneer=bool(_PANEER.search(t)),
        dried=bool(_DRIED.search(t)) or (bool(fruit_m) and not farm),
        fruit=("Mango" if fruit_m and fruit_m.group(1).lower().startswith("mango") else "Pineapple") if fruit_m else None,
        spoil=bool(_SPOIL.search(t)),
        both=bool(_BOTH.search(t)),
        cancel=bool(_CANCEL.search(t)),
        reset=bool(_RESET.search(t)),
        help=bool(_HELP.search(t)),
        vague=bool(_VAGUE.search(t)),
    )
    return i


# ------------------------------------------------------------------- state
@dataclass
class Workflow:
    task: Task
    key: str                      # farm | paneer | dried  (service being driven)
    fields: list[Field]
    idx: int = 0
    collected: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)   # optional fields left blank
    warnings: list[str] = field(default_factory=list)
    combined: bool = False
    pending_product: str | None = None   # combined: product named up-front
    pending_fruit: str | None = None


@dataclass
class ChatState:
    messages: list = field(default_factory=list)
    workflow: Workflow | None = None
    awaiting: dict | None = None         # {"kind": "intent"|"product", "combined": bool}
    results: dict = field(default_factory=dict)   # latest result per agent
    current_task: Task | None = None


WELCOME = (
    "Hello! I'm the **AgriChainAI** assistant. I coordinate two specialist agents:\n\n"
    "- 🌾 **Farm Agent** — recommends a crop with expected yield, profit and harvest timing\n"
    "- 🧀 **Spoilage Agent** — checks spoilage risk for **Paneer** and **dried Mango / Pineapple**\n\n"
    "Ask for a crop recommendation, a spoilage check, or run both analyses together. "
    "Use the quick actions above, or just type. I run offline with a rule-based router "
    "(no language model); say *cancel* at any time to stop a task."
)

CAPABILITIES = (
    "I can currently help with:\n\n"
    "- **Crop recommendation** — e.g. *“Suggest a profitable crop.”*\n"
    "- **Paneer spoilage** — e.g. *“Check whether my paneer is spoiled.”*\n"
    "- **Dried-fruit spoilage** (Mango, Pineapple) — e.g. *“Predict spoilage for my dried mango.”*\n"
    "- **Both analyses** — e.g. *“Run a farm recommendation and spoilage check.”*"
)


def _clip(s: str, n: int = 90) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class Orchestrator:
    def __init__(self, farm, paneer, dried):
        self.services = {"farm": farm, "paneer": paneer, "dried": dried}

    # ---------------------------------------------------------- public API
    def new_conversation(self, state: ChatState) -> None:
        state.messages.clear()
        state.workflow = None
        state.awaiting = None
        state.results.clear()
        state.current_task = None
        state.messages.append({"role": "assistant", "blocks": [{"type": "text", "text": WELCOME}], "trace": None})

    def statuses(self) -> dict:
        out = {}
        for label, key in (("Farm Agent", "farm"), ("Paneer Spoilage", "paneer"), ("Dried-Fruit Spoilage", "dried")):
            s = self.services[key]
            out[label] = (s.status, s.error)
        out["Orchestrator"] = ("Ready", None)
        return out

    def handle_message(self, state: ChatState, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        intent = detect_intent(text)
        if intent.reset:
            self.new_conversation(state)
            return
        trace: list = []
        self._step(trace, "User request received", "Supervisor", "OK", _clip(text))
        state.messages.append({"role": "user", "blocks": [{"type": "text", "text": text}], "trace": None})
        blocks: list = []
        try:
            self._route(state, text, intent, trace, blocks)
        except Exception as exc:  # last-resort guard: the chat must never crash
            self._step(trace, "Supervisor error", "Supervisor", "FAILED", f"{type(exc).__name__}: {exc}")
            blocks = [{"type": "error", "text": f"Something unexpected went wrong ({type(exc).__name__}: {exc}). "
                                                 "Your task was stopped; type a request to start again."}]
            state.workflow = state.awaiting = None
        self._step(trace, "Response formatted", "Supervisor", "OK")
        state.messages.append({"role": "assistant", "blocks": blocks, "trace": trace})

    # ------------------------------------------------------------- routing
    def _route(self, state, text, intent, trace, blocks):
        wf = state.workflow

        if intent.cancel and (wf or state.awaiting):
            self._step(trace, "Intent identified: CANCEL", "Supervisor", "OK", "keyword router")
            state.workflow = state.awaiting = None
            state.current_task = None
            blocks.append(self._t("Okay, I've cancelled that task and discarded the inputs collected so far. "
                                  "What would you like to do next?\n\n" + CAPABILITIES))
            return

        # -- answering a field of an active workflow
        if wf is not None and wf.idx < len(wf.fields):
            f = wf.fields[wf.idx]
            try:
                if f.optional and is_skip(text):
                    value, warns = None, []
                else:
                    value, warns = parse_value(f, text)
            except ValidationError as exc:
                other = intent.task
                if other is not None and other != self._task_of(wf):
                    self._step(trace, f"Intent identified: {other.value}", "Supervisor", "OK", "task switch requested")
                    state.workflow = state.awaiting = None
                    blocks.append(self._t(f"Switching tasks — discarding the unfinished {self._label(wf.task)} inputs."))
                    self._start(state, other, intent, trace, blocks)
                    return
                self._step(trace, f"Input validation ({f.label})", "Supervisor", "FAILED", str(exc))
                blocks.append(self._err(str(exc)))
                blocks.append(self._t(self._prompt(wf)))
                return
            self._step(trace, f"Input validation ({f.label})", "Supervisor", "PASSED",
                       "skipped (optional)" if value is None else str(value))
            wf.collected[f.key] = value
            if value is None:
                wf.skipped.append(f"{f.label}: {f.skip_note}")
            wf.warnings += warns
            wf.idx = self._next_missing(wf)
            if wf.idx < len(wf.fields):
                blocks.append(self._t(self._prompt(wf)))
            else:
                self._execute(state, trace, blocks)
            return

        # -- answering a clarification
        if state.awaiting:
            aw = state.awaiting
            task = intent.task
            if aw["kind"] == "product" and intent.product:
                state.awaiting = None
                self._step(trace, "Clarification resolved", "Supervisor", "OK", f"product = {intent.product}")
                self._begin_spoilage(state, intent.product, intent.fruit, aw.get("combined", False), trace, blocks)
                return
            if task is not None:   # a clear new request replaces a pending question
                state.awaiting = None
                self._step(trace, "Clarification resolved", "Supervisor", "OK", task.value)
                self._start(state, task, intent, trace, blocks)
                return
            if aw["kind"] == "intent" and intent.spoil:
                state.awaiting = None
                self._start(state, Task.CLARIFICATION_REQUIRED, intent, trace, blocks)
                return
            self._step(trace, "Intent identified: CLARIFICATION_REQUIRED", "Supervisor", "OK", "answer not understood")
            blocks.append(self._t(self._clarify_text(aw["kind"])))
            return

        # -- fresh request
        task = intent.task
        if task is None and intent.spoil:
            task = Task.CLARIFICATION_REQUIRED
        elif task is None and intent.vague:
            task = Task.CLARIFICATION_REQUIRED
        elif task is None and intent.help:
            self._step(trace, "Intent identified: HELP", "Supervisor", "OK", "keyword router")
            blocks.append(self._t(CAPABILITIES))
            return
        elif task is None:
            task = Task.UNSUPPORTED_REQUEST
        self._start(state, task, intent, trace, blocks)

    def _start(self, state, task, intent, trace, blocks):
        self._step(trace, f"Intent identified: {task.value}", "Supervisor", "OK", "deterministic keyword router")
        state.current_task = task
        if task is Task.UNSUPPORTED_REQUEST:
            blocks.append(self._t("I can't help with that yet. " + CAPABILITIES))
        elif task is Task.CLARIFICATION_REQUIRED:
            if intent.spoil and not intent.product:
                state.awaiting = {"kind": "product", "combined": False}
                blocks.append(self._t(self._clarify_text("product")))
            else:
                state.awaiting = {"kind": "intent"}
                blocks.append(self._t(self._clarify_text("intent")))
        elif task is Task.FARM_RECOMMENDATION:
            self._begin_farm(state, False, None, None, trace, blocks)
        elif task is Task.COMBINED_ANALYSIS:
            self._begin_farm(state, True, intent.product, intent.fruit, trace, blocks)
        else:
            key = "paneer" if task is Task.PANEER_SPOILAGE else "dried"
            self._begin_spoilage(state, key, intent.fruit, False, trace, blocks)

    def _begin_farm(self, state, combined, product, fruit, trace, blocks):
        svc = self.services["farm"]
        self._step(trace, "Supervisor selected: Farm Agent", "Supervisor", "OK", f"status={svc.status}")
        if svc.status != "Ready":
            blocks.append(self._unavailable("Farm Agent", svc))
            return
        wf = Workflow(Task.COMBINED_ANALYSIS if combined else Task.FARM_RECOMMENDATION, "farm",
                      svc.fields(), combined=combined, pending_product=product, pending_fruit=fruit)
        state.workflow = wf
        intro = ("Let's run **both analyses**. First, the **Farm Agent** needs your field conditions."
                 if combined else "Let's find a crop. The **Farm Agent** needs your field conditions.")
        blocks.append(self._t(intro + "\n\n" + self._prompt(wf)))

    def _begin_spoilage(self, state, key, fruit, combined, trace, blocks):
        svc = self.services[key]
        name = "Paneer Spoilage" if key == "paneer" else "Dried-Fruit Spoilage"
        self._step(trace, f"Supervisor selected: Spoilage Agent ({name})", "Supervisor", "OK", f"status={svc.status}")
        if svc.status != "Ready":
            blocks.append(self._unavailable(name, svc))
            return
        task = Task.PANEER_SPOILAGE if key == "paneer" else Task.DRIED_FRUIT_SPOILAGE
        wf = Workflow(Task.COMBINED_ANALYSIS if combined else task, key, svc.fields(), combined=combined)
        if key == "dried" and fruit:
            wf.collected["product_subtype"] = fruit       # stated by the user in their request
            self._step(trace, "Input taken from request", "Supervisor", "PASSED", f"product_subtype = {fruit}")
        wf.idx = self._next_missing(wf)
        state.workflow = wf
        state.current_task = task
        blocks.append(self._t(f"Now the **Spoilage Agent** ({name}).\n\n" + self._prompt(wf)))

    # ------------------------------------------------------------ execution
    def _execute(self, state, trace, blocks):
        wf = state.workflow
        svc = self.services[wf.key]
        self._step(trace, "Input validation: all fields", "Supervisor", "PASSED", f"{len(wf.collected)} values")
        summary = ", ".join(f"{k}={v}" for k, v in wf.collected.items() if v is not None)
        method = svc.api_name
        out, err = self._call(trace, svc.agent_name if wf.key == "farm" else "Spoilage Agent", method,
                              _clip(summary, 120), lambda: svc.recommend(wf.collected) if wf.key == "farm"
                              else svc.predict(wf.collected))
        warnings, skipped, combined, pending_product, pending_fruit = wf.warnings, wf.skipped, wf.combined, wf.pending_product, wf.pending_fruit
        state.workflow = None
        if err:
            blocks.append(self._err(err))
            blocks.append(self._t("The task was stopped. Check the message above, then ask again to retry."))
            return
        out["warnings"], out["defaults_used"] = warnings, skipped
        if wf.key == "farm":
            state.results["farm"] = out
            blocks.append({"type": "farm_result", "data": out})
            if combined:
                if pending_product:
                    self._begin_spoilage(state, pending_product, pending_fruit, True, trace, blocks)
                else:
                    state.awaiting = {"kind": "product", "combined": True}
                    blocks.append(self._t("Farm analysis complete. Next, the spoilage check — " + self._clarify_text("product")))
        else:
            state.results["spoilage"] = out
            blocks.append({"type": "spoilage_result", "data": out})
            if combined and "farm" in state.results:
                blocks.append({"type": "summary", "farm": state.results["farm"], "spoilage": out})
            blocks.append(self._t("Ask for another analysis any time."))

    def _call(self, trace, agent, api, summary, fn):
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000)
            self._step(trace, f"{api} executed", agent, "FAILED", f"{type(exc).__name__}: {exc}", ms)
            msg = str(exc) if type(exc).__name__ == "PredictionError" else (
                f"{agent} raised {type(exc).__name__}: {exc}")
            return None, msg
        self._step(trace, f"{api} executed", agent, "SUCCESS", summary, round((time.perf_counter() - t0) * 1000))
        return res, None

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _step(trace, step, agent, status, detail="", ms=None):
        trace.append({"step": step, "agent": agent, "status": status, "detail": detail, "duration_ms": ms})

    @staticmethod
    def _t(text):
        return {"type": "text", "text": text}

    @staticmethod
    def _err(text):
        return {"type": "error", "text": text}

    @staticmethod
    def _next_missing(wf):
        i = 0
        while i < len(wf.fields) and wf.fields[i].key in wf.collected:
            i += 1
        return i

    @staticmethod
    def _task_of(wf):
        return {"farm": Task.FARM_RECOMMENDATION, "paneer": Task.PANEER_SPOILAGE,
                "dried": Task.DRIED_FRUIT_SPOILAGE}[wf.key]

    @staticmethod
    def _label(task):
        return task.value.replace("_", " ").lower()

    @staticmethod
    def _prompt(wf):
        f = wf.fields[wf.idx]
        return f"**({wf.idx + 1}/{len(wf.fields)}) {f.label}** — {f.prompt}"

    @staticmethod
    def _unavailable(name, svc):
        why = svc.error or "its model artifacts are not available"
        return {"type": "error", "text": f"The {name} is not available ({svc.status}): {why}. "
                                         "Fix the issue (see the sidebar), then try again."}

    @staticmethod
    def _clarify_text(kind):
        if kind == "product":
            return ("Which stored product should I assess? Reply **Paneer**, **dried Mango** or **dried Pineapple**.")
        return ("I'm not sure which analysis you want. Do you want a **crop recommendation**, a **spoilage check** "
                "(Paneer or dried fruit), or **both**?")
