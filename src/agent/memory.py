from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .security import SecurityError, sanitize_memory_item, validate_user_id


class CustomerMemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def retrieve(self, user_id: str) -> dict[str, Any]:
        validate_user_id(user_id)
        data = self._load()
        return dict(data.get(user_id, {}))

    def update(self, user_id: str, memory_item: dict[str, Any]) -> dict[str, Any]:
        validate_user_id(user_id)
        sanitized = sanitize_memory_item(memory_item)
        data = self._load()
        profile = dict(data.get(user_id, {}))
        key = sanitized["key"]
        value = sanitized["value"]
        if key in {"repeated_issue_history", "unresolved_tickets"}:
            current = profile.get(key, [])
            if not isinstance(current, list):
                current = [current]
            current.append(value)
            profile[key] = current[-5:]
        else:
            profile[key] = value
        data[user_id] = profile
        self._save(data)
        return {"stored": {key: profile[key]}}


def describe_memory(memory: dict[str, Any]) -> str:
    if not memory:
        return "No stored customer preferences were found."
    parts: list[str] = []
    if memory.get("customer_tier"):
        parts.append(f"tier: {memory['customer_tier']}")
    if memory.get("preferred_contact"):
        parts.append(f"preferred contact: {memory['preferred_contact']}")
    if memory.get("shipping_preference"):
        parts.append(f"shipping preference: {memory['shipping_preference']}")
    if memory.get("refund_preference"):
        parts.append(f"refund preference: {memory['refund_preference']}")
    if memory.get("unresolved_tickets"):
        parts.append(f"unresolved tickets: {', '.join(memory['unresolved_tickets'])}")
    if memory.get("repeated_issue_history"):
        parts.append("repeated issue history noted")
    return "; ".join(parts) if parts else "Stored profile exists."
