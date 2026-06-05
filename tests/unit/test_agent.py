from __future__ import annotations

import json
from pathlib import Path

from agent.agent import CustomerSupportAgent, MockLLM


class RecordingLLM(MockLLM):
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def complete(self, prompt: str) -> str:
        self.tasks.append(prompt.splitlines()[0])
        return super().complete(prompt)


class BrokenPlanLLM(MockLLM):
    def complete(self, prompt: str) -> str:
        if prompt.startswith("TASK: create_action_plan"):
            return "not json"
        return super().complete(prompt)


class BadEscalationLLM(MockLLM):
    def complete(self, prompt: str) -> str:
        if prompt.startswith("TASK: decide_escalation"):
            return json.dumps({"should_escalate": False, "priority": "normal", "reason": "bad downgrade"})
        return super().complete(prompt)


def test_tool_routing_for_order_status(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is order ORD-1001?")

    assert result.tool_names == ["retrieve_customer_memory", "lookup_order"]
    assert "in_transit" in result.response


def test_tool_error_creates_retry_ticket(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is order ORD-9999?")

    assert "lookup_order" in result.tool_names
    assert "create_support_ticket" in result.tool_names
    assert "could not find ORD-9999" in result.response


def test_missing_order_id_clarification(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is my order?")

    assert "Please share the order ID" in result.response
    assert "lookup_order" not in result.tool_names


def test_multi_turn_missing_order_id_recovery(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", ["Where is my order?", "It is ORD-1001."])

    assert "lookup_order" in result.tool_names
    assert "Order ORD-1001" in result.response


def test_agent_calls_llm_for_understanding_planning_and_response(data_dir: Path) -> None:
    llm = RecordingLLM()
    agent = CustomerSupportAgent(mode="optimized", data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl", llm=llm)

    agent.handle("user_alex", "Where is order ORD-1001?")

    assert "TASK: understand_query" in llm.tasks
    assert "TASK: create_action_plan" in llm.tasks
    assert "TASK: decide_escalation" in llm.tasks
    assert "TASK: summarize_conversation" in llm.tasks
    assert "TASK: generate_response" in llm.tasks


def test_malformed_llm_plan_falls_back_to_tool_routing(data_dir: Path) -> None:
    agent = CustomerSupportAgent(mode="optimized", data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl", llm=BrokenPlanLLM())

    result = agent.handle("user_alex", "Where is order ORD-1001?")

    assert "lookup_order" in result.tool_names
    assert "Order ORD-1001" in result.response


def test_llm_cannot_downgrade_mandatory_escalation(data_dir: Path) -> None:
    agent = CustomerSupportAgent(mode="optimized", data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl", llm=BadEscalationLLM())

    result = agent.handle("user_alex", "Refund ORD-1002 or I will file a chargeback.")

    assert result.should_escalate is True
    assert "escalate_to_human" in result.tool_names


def test_final_summary_is_separated_when_user_asks_to_summarize(optimized_agent) -> None:
    result = optimized_agent.handle(
        "user_alex",
        [
            "Where is my order?",
            "It is ORD-1001.",
            "Can you tell me the shipping status?",
            "Thanks. Summarize what happened and what tools were used.",
        ],
    )

    assert "Conversation summary:" in result.response
    assert "Tools used:" in result.response
