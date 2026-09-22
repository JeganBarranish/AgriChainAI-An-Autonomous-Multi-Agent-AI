"""
AgriChainAI multi-product spoilage prediction — interactive CLI.

[RULE-BASED UI + SUPERVISED model inference]

Usage:
    python -m spoilage.inference
    python -m spoilage.inference --train   # train then infer
"""

from __future__ import annotations

import argparse
import sys

from spoilage.explainability import DRYFRUIT_DISPLAY, PANEER_DISPLAY, format_explanation
from spoilage.predictor import PredictionError, predict_dryfruit, predict_paneer


def _ask(prompt: str, error_msg: str, cast, check):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "":
                raise ValueError("empty")
            value = cast(raw)
            if check(value):
                return value
        except (ValueError, TypeError):
            pass
        print(error_msg)


def _menu(prompt: str, options: dict[int, str]) -> int:
    for k, v in options.items():
        print(f"{k}. {v}")
    while True:
        try:
            choice = int(input(prompt).strip())
            if choice in options:
                return choice
        except ValueError:
            pass
        print(f"Invalid choice. Enter one of: {list(options.keys())}")


def _explain_paneer(result: dict) -> str:
    factors = []
    if result["storage_duration_days"] >= 6:
        factors.append("prolonged storage duration")
    if result["temperature"] > 4.0:
        factors.append("elevated storage temperature")
    if result["humidity"] > 50:
        factors.append("elevated humidity conditions")
    if not factors:
        factors.append("cold-chain conditions within typical training ranges")
    return "Main contributing conditions:\n" + "\n".join(f"- {f}" for f in factors)


def _explain_dryfruit(result: dict) -> str:
    imp = result.get("feature_importance", {})
    display = result.get("display_map", DRYFRUIT_DISPLAY)
    bullets = format_explanation(imp, display, top_k=4)
    lines = ["Main contributing conditions (from model feature importance):"]
    lines.extend(bullets)
    if result["storage_duration_days"] >= 90:
        lines.append("- extended storage duration (≥ 3 months equivalent)")
    return "\n".join(lines)


def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("                 SPOILAGE ANALYSIS")
    print("=" * 60)
    print()
    print(f"Product              : {result['product_subtype']}")

    if result["product_type"] == "paneer":
        print(f"Storage Duration     : {result['storage_duration_days']} days")
        print(f"Temperature          : {result['temperature']} °C")
        print(f"Humidity             : {result['humidity']} %")
    else:
        print(f"Storage Duration     : {result['storage_duration_days']} days")
        print(f"Dryer Type           : {result['dryer_type']}")
        print(f"Packaging            : {result['packaging']}")
        print("  (Note: temperature/RH not in Mendeley dataset — not used)")

    print()
    print(f"Fresh Probability     : {result['fresh_probability'] * 100:>5.1f} %")
    print(f"Warning Probability   : {result['warning_probability'] * 100:>5.1f} %")
    print(f"Spoiled Probability   : {result['spoiled_probability'] * 100:>5.1f} %")
    print()
    print(f"Predicted Status      : {result['status']}")
    print(f"Spoilage Risk         : {result['spoilage_risk']:>5.1f} %")
    print()
    print("-" * 60)
    print("Interpretation:")
    if result["product_type"] == "paneer":
        print(_explain_paneer(result))
    else:
        print(_explain_dryfruit(result))
    print(f"\nModel: {result['model_used']} | Confidence: {result['confidence'] * 100:.1f} %")
    print("=" * 60)


def run_paneer_flow() -> None:
    print("\n" + "-" * 60)
    print("                  PANEER ANALYSIS")
    print("-" * 60)
    print("\nEnter cold-chain conditions.")
    print("Compressor activity & fridge openings are optional — press Enter")
    print("to use training-data medians (recommended when sensor history unavailable).\n")

    storage = _ask(
        "Enter Storage Duration (days): ",
        "Invalid: enter a number >= 0.",
        float, lambda v: v >= 0,
    )
    temperature = _ask(
        "Enter Temperature (°C): ",
        "Invalid: enter -30 to 40.",
        float, lambda v: -30 <= v <= 40,
    )
    humidity = _ask(
        "Enter Humidity (%): ",
        "Invalid: enter 0 to 100.",
        float, lambda v: 0 <= v <= 100,
    )

    comp_raw = input("Enter Compressor Activity (%, optional): ").strip()
    fridge_raw = input("Enter Fridge Opening Count (optional): ").strip()
    kwargs = dict(
        temperature=temperature, humidity=humidity, storage_duration=storage,
    )
    if comp_raw:
        val = float(comp_raw)
        if not (0 <= val <= 100):
            raise PredictionError("Compressor activity must be 0–100 %.")
        kwargs["compressor_activity"] = val
    if fridge_raw:
        val = int(fridge_raw)
        if val < 0:
            raise PredictionError("Fridge opening count must be >= 0.")
        kwargs["fridge_opening_count"] = val

    print("\nPredicting...")
    result = predict_paneer(**kwargs)
    _print_result(result)


def run_dryfruit_flow() -> None:
    print("\n" + "-" * 60)
    print("                  DRIED FRUIT ANALYSIS")
    print("-" * 60)
    choice = _menu("\nSelect Product:\n", {1: "Mango", 2: "Pineapple"})
    fruit = "Mango" if choice == 1 else "Pineapple"

    storage = _ask(
        "Enter Storage Duration (days): ",
        "Invalid: enter a number >= 0.",
        float, lambda v: v >= 0,
    )
    print("\nDryer type: CMD = cabinet mixed-mode solar dryer, TD = tunnel dryer")
    dryer = _ask(
        "Enter Dryer Type (CMD/TD): ",
        "Invalid: enter CMD or TD.",
        str, lambda v: v.strip().upper() in ("CMD", "TD", "MIXED", "TUNNEL", "TUNEL"),
    )
    print("\nPackaging: None, HDPE (high density), LDPE (low density)")
    packaging = _ask(
        "Enter Packaging Condition (None/HDPE/LDPE): ",
        "Invalid: enter None, HDPE, or LDPE.",
        str, lambda v: v.strip().upper() in ("NONE", "NO", "HDPE", "HIGH DENSITY", "LDPE", "LOW DENSITY"),
    )

    print("\nPredicting...")
    result = predict_dryfruit(
        product_subtype=fruit,
        storage_duration_days=storage,
        dryer_type=dryer,
        packaging=packaging,
    )
    _print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriChainAI Spoilage Prediction")
    parser.add_argument("--train", action="store_true", help="Train models before inference")
    args = parser.parse_args()

    if args.train:
        from spoilage.trainer import train_all
        train_all()

    print("\n" + "=" * 60)
    print("             AgriChainAI Spoilage Prediction")
    print("=" * 60)
    print("\nSelect Product Type\n")
    choice = _menu("", {1: "Paneer", 2: "Dried Fruit"})

    try:
        if choice == 1:
            run_paneer_flow()
        else:
            run_dryfruit_flow()
    except PredictionError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
