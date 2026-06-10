"""Ask JISR a question.

Usage:
    python -m src.agents.run --query "ما هي شروط الدفع في العقد؟"
"""
from __future__ import annotations

import argparse

from src.agents.graph import answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    state = answer(args.query)
    print(f"\nLang routed : {state.get('lang')}")
    print(f"Reflections : {state.get('loops')}")
    print(f"Critic      : {state.get('critique')}")
    print(f"\nAnswer:\n{state.get('answer')}\n")


if __name__ == "__main__":
    main()
