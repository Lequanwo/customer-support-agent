from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import CustomerSupportAgent  # noqa: E402


def print_turn(role: str, text: str) -> None:
    print(f"{role}: {text}")
    print()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        shutil.copytree(ROOT / "data", data_dir)
        (data_dir / "audit_log.jsonl").write_text("", encoding="utf-8")

        agent = CustomerSupportAgent(
            mode="optimized",
            data_dir=data_dir,
            audit_path=data_dir / "audit_log.jsonl",
        )

        user_id = "user_alex"
        transcript: list[str] = []
        turns = [
            "Hi, I need help with an order.",
            "I do not have the order number in front of me.",
            "Actually, I found it. It is ORD-1001.",
            "Can you tell me the shipping status?",
            "Also, please remember my preferred contact method is phone.",
            "Do you still remember my shipping preference?",
            "Now I have another question about order ORD-1002.",
            "Can I get a refund for ORD-1002?",
            "The coffee grinder also arrived damaged.",
            "I am frustrated, but I just want this solved.",
            "Please create whatever support record is needed.",
            "Thanks. Summarize what happened and what tools were used.",
        ]

        print("Long multi-turn customer support demo")
        print("=" * 42)
        print()

        for index, user_text in enumerate(turns, start=1):
            transcript.append(user_text)
            result = agent.handle(user_id, transcript)
            print_turn(f"Turn {index} Customer", user_text)
            print_turn("Agent", result.response)
            print(f"Tools this turn: {result.tool_names or ['none']}")
            print("-" * 42)
            print()

        audit_count = len(
            [line for line in (data_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        )
        print(f"Audit log entries created during demo: {audit_count}")


if __name__ == "__main__":
    main()
