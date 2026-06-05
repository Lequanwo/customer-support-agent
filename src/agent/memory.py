from __future__ import annotations

import json
import os
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


class Mem0MemoryStore:
    """Optional Mem0-backed memory store.

    This adapter keeps the project API the same as the local JSON store. It is
    only used when MEMORY_BACKEND=mem0, so the default eval remains
    deterministic and offline.
    """

    def __init__(self, api_key: str | None = None):
        try:
            from mem0 import MemoryClient
        except ImportError as exc:
            raise RuntimeError("Install mem0ai to use MEMORY_BACKEND=mem0: python -m pip install mem0ai") from exc
        self.client = MemoryClient(api_key=api_key or os.getenv("MEM0_API_KEY"))

    def retrieve(self, user_id: str) -> dict[str, Any]:
        validate_user_id(user_id)
        query = (
            "customer tier, preferred contact method, repeated issue history, "
            "previous unresolved tickets, refund preference, shipping preference"
        )
        try:
            results = self.client.search(query, user_id=user_id)
        except TypeError:
            results = self.client.search(query, filters={"user_id": user_id})
        return {"mem0_results": self._normalize_search_results(results)}

    def update(self, user_id: str, memory_item: dict[str, Any]) -> dict[str, Any]:
        validate_user_id(user_id)
        sanitized = sanitize_memory_item(memory_item)
        memory_text = f"{sanitized['key']}: {sanitized['value']}"
        try:
            self.client.add(memory_text, user_id=user_id)
        except TypeError:
            self.client.add([{"role": "user", "content": memory_text}], user_id=user_id)
        return {"stored": {sanitized["key"]: sanitized["value"]}}

    @staticmethod
    def _normalize_search_results(results: Any) -> list[str]:
        if isinstance(results, dict):
            candidates = results.get("results") or results.get("memories") or []
        else:
            candidates = results or []
        normalized: list[str] = []
        for item in candidates:
            if isinstance(item, dict):
                text = item.get("memory") or item.get("text") or item.get("content")
                if text:
                    normalized.append(str(text))
            elif item:
                normalized.append(str(item))
        return normalized[:10]


def make_memory_store(path: str | Path):
    backend = os.getenv("MEMORY_BACKEND", "json").lower()
    if backend == "json":
        return CustomerMemoryStore(path)
    if backend == "mem0":
        return Mem0MemoryStore()
    raise RuntimeError("MEMORY_BACKEND must be 'json' or 'mem0'.")


def describe_memory(memory: dict[str, Any]) -> str:
    if not memory:
        return "No stored customer preferences were found."
    if memory.get("mem0_results"):
        return "Mem0 memories: " + "; ".join(memory["mem0_results"])
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
