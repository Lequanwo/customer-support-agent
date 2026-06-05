from __future__ import annotations

from pathlib import Path

from agent.tools import SupportTools


def test_lookup_order_routes_and_audits(data_dir: Path) -> None:
    calls = []
    tools = SupportTools(data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl")
    result = tools.lookup_order(calls, "ORD-1001", "user_alex")

    assert result["status"] == "in_transit"
    assert calls[0].name == "lookup_order"
    assert (data_dir / "audit_log.jsonl").read_text(encoding="utf-8").strip()


def test_unauthorized_order_lookup_blocked(data_dir: Path) -> None:
    calls = []
    tools = SupportTools(data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl")
    result = tools.lookup_order(calls, "ORD-1003", "user_alex")

    assert result["security_violation"] is True
    assert "does not belong" in result["error"]
