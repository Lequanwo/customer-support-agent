from __future__ import annotations

import argparse

from .agent import CustomerSupportAgent, make_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic customer support agent.")
    parser.add_argument("--user-id", default="user_alex")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--llm", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default=None, help="OpenAI model to use when --llm openai is selected.")
    parser.add_argument("message", nargs="+")
    args = parser.parse_args()
    agent = CustomerSupportAgent(mode=args.mode, llm=make_llm(args.llm, model=args.model))
    result = agent.handle(args.user_id, " ".join(args.message))
    print(result.response)
    if result.tool_names:
        print("Tools:", ", ".join(result.tool_names))


if __name__ == "__main__":
    main()
