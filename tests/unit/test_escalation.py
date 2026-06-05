from __future__ import annotations


def test_escalates_when_user_asks_for_human(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "I want a human representative.")

    assert result.should_escalate is True
    assert "escalate_to_human" in result.tool_names


def test_escalates_for_chargeback_threat(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Refund ORD-1002 or I will file a chargeback.")

    assert result.should_escalate is True
    assert result.escalations[0]["priority"] in {"high", "urgent"}
