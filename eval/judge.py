from __future__ import annotations

from typing import Any


def judge_a(case: dict[str, Any], actual: dict[str, Any]) -> int:
    """Heuristic judge: response passes if it has expected facts and correct escalation."""
    text = actual["response"].lower()
    facts_ok = all(fact.lower() in text for fact in case["expected_facts"])
    forbidden_ok = not any(fact.lower() in text for fact in case["forbidden_facts"])
    escalation_ok = actual["should_escalate"] == case["should_escalate"]
    return int(facts_ok and forbidden_ok and escalation_ok)


def judge_b(case: dict[str, Any], actual: dict[str, Any]) -> int:
    """Mock second judge: deterministic rubric focused on policy and tool behavior."""
    text = actual["response"].lower()
    tools = actual["tool_names"]
    expected_tool_hits = sum(1 for tool in case["expected_tools"] if tool in tools)
    tool_ok = expected_tool_hits >= max(1, len(case["expected_tools"]) - 1)
    forbidden_tool_ok = not any(tool in tools for tool in case["forbidden_tools"])
    response_ok = all(fact.lower() in text for fact in case["expected_facts"])
    forbidden_fact_ok = not any(fact.lower() in text for fact in case["forbidden_facts"])
    memory_ok = actual["memory_reads"] >= case["expected_memory_reads"]
    escalation_ok = actual["should_escalate"] == case["should_escalate"]
    return int(tool_ok and forbidden_tool_ok and response_ok and forbidden_fact_ok and memory_ok and escalation_ok)


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have the same length.")
    n = len(labels_a)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_yes_a = sum(labels_a) / n
    p_yes_b = sum(labels_b) / n
    p_no_a = 1 - p_yes_a
    p_no_b = 1 - p_yes_b
    expected = p_yes_a * p_yes_b + p_no_a * p_no_b
    if expected == 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def score_quality(cases: list[dict[str, Any]], actuals: list[dict[str, Any]]) -> dict[str, Any]:
    labels_a = [judge_a(case, actual) for case, actual in zip(cases, actuals)]
    labels_b = [judge_b(case, actual) for case, actual in zip(cases, actuals)]
    return {
        "helpfulness": sum(labels_a) / max(1, len(labels_a)),
        "policy_correctness": sum(labels_b) / max(1, len(labels_b)),
        "escalation_appropriateness": sum(
            actual["should_escalate"] == case["should_escalate"] for case, actual in zip(cases, actuals)
        )
        / max(1, len(cases)),
        "memory_usage_correctness": sum(
            actual["memory_reads"] >= case["expected_memory_reads"] for case, actual in zip(cases, actuals)
        )
        / max(1, len(cases)),
        "cohens_kappa": cohens_kappa(labels_a, labels_b),
    }
