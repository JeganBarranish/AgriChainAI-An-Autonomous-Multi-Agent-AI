"""
Input-field definitions and validation shared by the adapters and the
orchestrator. Pure Python (no Streamlit / model imports) so it is trivially
testable.

Hard limits here are UI sanity checks (physically implausible values), NOT
properties of the trained models. Where the model's own training data covers a
narrower range, a Field carries a `soft_range`: values outside it are accepted
but flagged, because the model is then extrapolating.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


class ValidationError(ValueError):
    """User input rejected; the message explains how to fix it."""


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str                       # float | int | choice | bool | date
    prompt: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    aliases: dict = field(default_factory=dict)   # lowercase text -> canonical choice
    optional: bool = False
    skip_note: str = ""             # what happens when an optional field is skipped
    unit: str = ""
    soft_range: tuple | None = None  # (lo, hi) observed in the model's training data


SKIP_WORDS = {"skip", "default", "n/a", "na"}
_NUM = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?|c|%|percent|days?|d|mm|ha|hectares?)?\s*$",
    re.IGNORECASE,
)
_YES = {"y", "yes", "true", "1"}
_NO = {"n", "no", "false", "0"}


def is_skip(text: str) -> bool:
    return text.strip().lower() in SKIP_WORDS


def _range_text(f: Field) -> str:
    lo = "" if f.minimum is None else f"{f.minimum:g}"
    hi = "" if f.maximum is None else f"{f.maximum:g}"
    return f"{lo} to {hi}{(' ' + f.unit) if f.unit else ''}".strip()


def parse_value(f: Field, text: str, today: date | None = None):
    """Return (value, warnings). Raise ValidationError on bad input."""
    raw = (text or "").strip()
    if not raw:
        raise ValidationError(f"Please enter a value for {f.label}.")
    warnings: list[str] = []

    if f.kind in ("float", "int"):
        m = _NUM.match(raw)
        if not m:
            raise ValidationError(
                f"'{raw}' is not a valid number for {f.label}. "
                f"Enter a number between {_range_text(f)} (for example 12.5).")
        value = float(m.group(1))
        if not math.isfinite(value):
            raise ValidationError(f"{f.label} must be a finite number.")
        if f.kind == "int":
            if value != int(value):
                raise ValidationError(f"{f.label} must be a whole number.")
            value = int(value)
        if f.minimum is not None and value < f.minimum or f.maximum is not None and value > f.maximum:
            raise ValidationError(f"{f.label} must be between {_range_text(f)}; you entered {value:g}.")
        if f.soft_range and not (f.soft_range[0] <= value <= f.soft_range[1]):
            warnings.append(
                f"{f.label} = {value:g} is outside the range the model was trained on "
                f"({f.soft_range[0]:g}–{f.soft_range[1]:g}); the prediction is an extrapolation.")
        return value, warnings

    if f.kind == "choice":
        key = raw.lower()
        for c in f.choices:
            if key == str(c).lower():
                return c, warnings
        if key in f.aliases:
            return f.aliases[key], warnings
        raise ValidationError(f"'{raw}' is not a valid {f.label}. Choose one of: {', '.join(map(str, f.choices))}.")

    if f.kind == "bool":
        key = raw.lower()
        if key in _YES:
            return True, warnings
        if key in _NO:
            return False, warnings
        raise ValidationError(f"Please answer yes or no for {f.label}.")

    if f.kind == "date":
        today = today or date.today()
        if raw.lower() == "today":
            return today, warnings
        try:
            d = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            raise ValidationError(f"'{raw}' is not a valid date. Use YYYY-MM-DD (for example {today.isoformat()}) or type 'today'.")
        if not (today - timedelta(days=365) <= d <= today + timedelta(days=730)):
            raise ValidationError(f"{f.label} must be within one year before and two years after today.")
        return d, warnings

    raise ValidationError(f"Unsupported field type '{f.kind}' for {f.label}.")
