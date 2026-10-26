from datetime import date

import pytest

from agents.fields import Field, ValidationError, is_skip, parse_value

TEMP = Field("temperature", "Temperature", "float", "?", minimum=-30, maximum=40, unit="°C", soft_range=(3.7, 4.4))
COUNT = Field("n", "Fridge opening count", "int", "?", minimum=0, maximum=1000)
DRYER = Field("dryer", "Dryer type", "choice", "?", choices=("CMD", "TD"), aliases={"tunnel": "TD"})
YN = Field("irrigation", "Irrigation", "bool", "?")
DATE = Field("sowing", "Sowing date", "date", "?")


def test_numbers_with_units():
    assert parse_value(TEMP, "4")[0] == 4.0
    assert parse_value(TEMP, " 4.5 °C ")[0] == 4.5
    assert parse_value(TEMP, "-5")[0] == -5.0


@pytest.mark.parametrize("bad", ["abc", "", "4..5", "nan", "inf", "12 kg", "41", "-31"])
def test_bad_numbers_rejected(bad):
    with pytest.raises(ValidationError):
        parse_value(TEMP, bad)


def test_soft_range_warns_but_accepts():
    value, warns = parse_value(TEMP, "6")
    assert value == 6.0 and len(warns) == 1 and "extrapolation" in warns[0]
    assert parse_value(TEMP, "4.0")[1] == []


def test_int_rejects_fraction():
    assert parse_value(COUNT, "3")[0] == 3
    with pytest.raises(ValidationError):
        parse_value(COUNT, "3.5")
    with pytest.raises(ValidationError):
        parse_value(COUNT, "-1")


def test_choice_and_alias():
    assert parse_value(DRYER, "cmd")[0] == "CMD"
    assert parse_value(DRYER, "Tunnel")[0] == "TD"
    with pytest.raises(ValidationError):
        parse_value(DRYER, "oven")


def test_bool():
    assert parse_value(YN, "Yes")[0] is True
    assert parse_value(YN, "no")[0] is False
    with pytest.raises(ValidationError):
        parse_value(YN, "maybe")


def test_date():
    today = date(2026, 10, 20)
    assert parse_value(DATE, "today", today)[0] == today
    assert parse_value(DATE, "2026-11-01", today)[0] == date(2026, 11, 1)
    for bad in ("01/11/2026", "2026-13-01", "2035-01-01"):
        with pytest.raises(ValidationError):
            parse_value(DATE, bad, today)


def test_skip_words():
    assert is_skip("Skip") and is_skip("default") and not is_skip("None")
