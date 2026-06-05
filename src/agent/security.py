from __future__ import annotations

import re
from typing import Any

SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # payment card-like
    re.compile(r"\bpassword\b|\bpasscode\b|\bpin\b", re.IGNORECASE),
    re.compile(r"\bdiagnosis\b|\bmedical\b|\battorney\b|\blegal advice\b", re.IGNORECASE),
]

ALLOWED_MEMORY_KEYS = {
    "customer_tier",
    "preferred_contact",
    "repeated_issue_history",
    "unresolved_tickets",
    "refund_preference",
    "shipping_preference",
}


class SecurityError(ValueError):
    pass


def validate_user_id(user_id: str) -> str:
    if not re.fullmatch(r"user_[a-z0-9_]+", user_id or ""):
        raise SecurityError("Invalid user_id.")
    return user_id


def validate_order_id(order_id: str) -> str:
    if not re.fullmatch(r"ORD-\d{4}", order_id or ""):
        raise SecurityError("Invalid order_id.")
    return order_id


def validate_priority(priority: str) -> str:
    normalized = (priority or "").lower()
    if normalized not in {"low", "normal", "high", "urgent"}:
        raise SecurityError("Invalid priority.")
    return normalized


def contains_sensitive_info(value: Any) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def sanitize_memory_item(memory_item: dict[str, Any]) -> dict[str, Any]:
    key = str(memory_item.get("key", "")).strip()
    value = memory_item.get("value")
    if key not in ALLOWED_MEMORY_KEYS:
        raise SecurityError(f"Memory key '{key}' is not allowed.")
    if contains_sensitive_info(value):
        raise SecurityError("Sensitive information cannot be stored in memory.")
    if isinstance(value, str):
        value = value.strip()[:300]
    return {"key": key, "value": value}


def assert_order_belongs_to_user(order: dict[str, Any], user_id: str) -> None:
    if order.get("user_id") != user_id:
        raise SecurityError("Order does not belong to this user.")


def high_risk_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(chargeback|sue|lawsuit|lawyer|attorney|fraud|unsafe|injured|medical|hospital)\b",
            text,
            re.IGNORECASE,
        )
    )


def explicit_human_request(text: str) -> bool:
    return bool(re.search(r"\b(human|manager|supervisor|representative|agent)\b", text, re.IGNORECASE))


def very_angry(text: str) -> bool:
    return bool(re.search(r"\b(furious|outraged|unacceptable|never buying|worst|scam)\b", text, re.IGNORECASE))
