from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import CustomerSupportAgent  # noqa: E402
from judge import score_quality  # noqa: E402
from metrics import score_case, summarize  # noqa: E402


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        shutil.copytree(ROOT / "data", data_dir)
        audit_path = data_dir / "audit_log.jsonl"
        audit_path.write_text("", encoding="utf-8")
        agent = CustomerSupportAgent(mode=mode, data_dir=data_dir, audit_path=audit_path)
        result = agent.handle(case["user_id"], case["input"])
        audit_log_count = len([line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        return {
            "id": case["id"],
            "user_id": case["user_id"],
            "response": result.response,
            "tool_names": result.tool_names,
            "tool_calls": [
                {"name": call.name, "ok": call.ok, "args": call.args, "result": call.result}
                for call in result.tool_calls
            ],
            "should_escalate": result.should_escalate,
            "memory_reads": result.memory_reads,
            "memory_writes": result.memory_writes,
            "security_violations": result.security_violations,
            "audit_log_count": audit_log_count,
            "escalations": result.escalations,
        }


def write_markdown_summary(mode: str, summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        f"# {mode.title()} Eval Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Overall score | {summary['overall_score']:.4f} |",
        f"| Pass rate | {summary['pass_rate']:.4f} |",
        f"| Memory retention accuracy | {summary['memory_retention_accuracy']:.4f} |",
        f"| Escalation decision accuracy | {summary['escalation_decision_accuracy']:.4f} |",
        f"| Tool routing accuracy | {summary['tool_routing_accuracy']:.4f} |",
        f"| Expected fact recall | {summary['expected_fact_recall']:.4f} |",
        f"| Forbidden tool violation rate | {summary['forbidden_tool_violation_rate']:.4f} |",
        f"| Forbidden fact violation rate | {summary['forbidden_fact_violation_rate']:.4f} |",
        f"| Security violation count | {summary['security_violation_count']} |",
        f"| Judge Cohen's kappa | {summary['judge']['cohens_kappa']:.4f} |",
        "",
        "## Failed Cases",
        "",
    ]
    lines.extend(f"- {case_id}" for case_id in summary["failed_cases"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    args = parser.parse_args()

    cases = load_golden(ROOT / "eval" / "golden.jsonl")
    actuals = [run_case(case, args.mode) for case in cases]
    scored = [score_case(case, actual) for case, actual in zip(cases, actuals)]
    judge_summary = score_quality(cases, actuals)
    summary = summarize(scored, judge_summary)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mode": args.mode, "summary": summary, "cases": scored, "actuals": actuals}
    json_path = out_dir / f"{args.mode}_results.json"
    md_path = out_dir / f"{args.mode}_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_summary(args.mode, summary, md_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
