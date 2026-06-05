from __future__ import annotations

from dataclasses import dataclass

from .security import explicit_human_request, high_risk_text, very_angry


@dataclass
class EscalationDecision:
    should_escalate: bool
    priority: str
    reason: str


def decide_escalation(text: str, *, repeated_issue: bool = False, ambiguous_policy: bool = False, low_confidence: bool = False) -> EscalationDecision:
    lowered = text.lower()
    if high_risk_text(text):
        return EscalationDecision(True, "high", "High-risk, legal, chargeback, fraud, or safety language.")
    if explicit_human_request(text):
        return EscalationDecision(True, "normal", "Customer explicitly asked for a human.")
    if ambiguous_policy:
        return EscalationDecision(True, "normal", "Refund policy is ambiguous for this order.")
    if repeated_issue or "third time" in lowered or "again and again" in lowered:
        return EscalationDecision(True, "high", "Repeated unresolved issue.")
    if very_angry(text) and ("refund" in lowered or "damaged" in lowered):
        return EscalationDecision(False, "normal", "Customer is angry, but the issue appears solvable with available tools.")
    if low_confidence:
        return EscalationDecision(True, "normal", "Low confidence automated resolution.")
    return EscalationDecision(False, "normal", "Automated tools can handle this request.")
