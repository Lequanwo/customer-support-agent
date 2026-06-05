from __future__ import annotations

import shutil
import sys
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import CustomerSupportAgent, make_llm  # noqa: E402


def print_turn(role: str, text: str) -> None:
    print(f"{role}: {text}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 12-turn customer support demo.")
    parser.add_argument("--llm", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default=None, help="OpenAI model to use when --llm openai is selected.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        shutil.copytree(ROOT / "data", data_dir)
        (data_dir / "audit_log.jsonl").write_text("", encoding="utf-8")

        agent = CustomerSupportAgent(
            mode="optimized",
            data_dir=data_dir,
            audit_path=data_dir / "audit_log.jsonl",
            llm=make_llm(args.llm, model=args.model),
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
        final_summary = ""
        final_tools: list[str] = []

        for index, user_text in enumerate(turns, start=1):
            transcript.append(user_text)
            result = agent.handle(user_id, transcript)
            display_response = result.response
            if index == len(turns) and "Conversation summary:" in result.response:
                display_response, final_summary = result.response.split("Conversation summary:", 1)
                display_response = display_response.strip()
                final_summary = final_summary.strip()
                if "Tools used:" in final_summary:
                    final_summary, tools_text = final_summary.split("Tools used:", 1)
                    final_summary = final_summary.strip()
                    final_tools = [tool.strip().strip(".") for tool in tools_text.split(",") if tool.strip()]
                else:
                    final_tools = result.tool_names

            print_turn(f"Turn {index} Customer", user_text)
            print_turn("Agent", display_response)
            print(f"Tools this turn: {result.tool_names or ['none']}")
            print("-" * 42)
            print()

        audit_count = len(
            [line for line in (data_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        )
        if final_summary:
            print("Final Summary")
            print("=" * 42)
            print(final_summary)
            print()
            print(f"Tools used: {final_tools}")
            print()
        print(f"Audit log entries created during demo: {audit_count}")


if __name__ == "__main__":
    main()
