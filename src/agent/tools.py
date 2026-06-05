from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .audit import AuditLogger
from .memory import make_memory_store
from .models import ToolCall
from .security import (
    SecurityError,
    assert_order_belongs_to_user,
    validate_order_id,
    validate_priority,
    validate_user_id,
)


class SupportTools:
    def __init__(self, data_dir: str | Path = "data", audit_path: str | Path = "data/audit_log.jsonl"):
        self.data_dir = Path(data_dir)
        self.audit = AuditLogger(audit_path)
        self.memory = make_memory_store(self.data_dir / "memory_store.json")
        self._ticket_counter = 3000

    def _read_json(self, name: str) -> Any:
        return json.loads((self.data_dir / name).read_text(encoding="utf-8"))

    def _record(
        self,
        calls: list[ToolCall],
        name: str,
        args: dict[str, Any],
        func: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        user_id = str(args.get("user_id", "unknown"))
        try:
            result = func()
            ok = True
        except SecurityError as exc:
            result = {"error": str(exc), "security_violation": True}
            ok = False
        except Exception as exc:  # deterministic local tool errors still need audit.
            result = {"error": str(exc), "security_violation": False}
            ok = False
        self.audit.log(tool_name=name, user_id=user_id, args=args, ok=ok, result=result)
        calls.append(ToolCall(name=name, args=args, ok=ok, result=result))
        return result

    def lookup_order(self, calls: list[ToolCall], order_id: str, user_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            validate_user_id(user_id)
            validate_order_id(order_id)
            orders = self._read_json("orders.json")
            order = orders.get(order_id)
            if not order:
                return {"error": "Order not found.", "retryable": True}
            assert_order_belongs_to_user(order, user_id)
            return {"order": order_id, **order}

        return self._record(calls, "lookup_order", {"order_id": order_id, "user_id": user_id}, run)

    def check_refund_policy(self, calls: list[ToolCall], order_id: str, user_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            validate_user_id(user_id)
            validate_order_id(order_id)
            orders = self._read_json("orders.json")
            policies = self._read_json("policies.json")
            order = orders.get(order_id)
            if not order:
                return {"error": "Order not found.", "retryable": True}
            assert_order_belongs_to_user(order, user_id)
            if order.get("ambiguous_policy"):
                return {
                    "eligible": False,
                    "ambiguous": True,
                    "reason": "Policy needs human review because the order has an exception flag.",
                }
            if order.get("final_sale"):
                return {"eligible": False, "ambiguous": False, "reason": "Final sale items are not refundable."}
            days = int(order.get("days_since_delivery", 999))
            window = int(policies["refund_window_days"])
            if order.get("status") == "delivered" and days <= window:
                return {"eligible": True, "ambiguous": False, "reason": f"Delivered {days} days ago within {window}-day window."}
            return {"eligible": False, "ambiguous": False, "reason": f"Outside the {window}-day refund window or not delivered."}

        return self._record(calls, "check_refund_policy", {"order_id": order_id, "user_id": user_id}, run)

    def create_support_ticket(self, calls: list[ToolCall], user_id: str, issue_summary: str, priority: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            validate_user_id(user_id)
            validated_priority = validate_priority(priority)
            self._ticket_counter += 1
            return {
                "ticket_id": f"TCK-{self._ticket_counter}",
                "issue_summary": issue_summary[:300],
                "priority": validated_priority,
            }

        return self._record(
            calls,
            "create_support_ticket",
            {"user_id": user_id, "issue_summary": issue_summary, "priority": priority},
            run,
        )

    def escalate_to_human(
        self,
        calls: list[ToolCall],
        user_id: str,
        reason: str,
        conversation_summary: str,
        priority: str,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            validate_user_id(user_id)
            validated_priority = validate_priority(priority)
            self._ticket_counter += 1
            return {
                "escalation_id": f"ESC-{self._ticket_counter}",
                "reason": reason[:200],
                "conversation_summary": conversation_summary[:500],
                "priority": validated_priority,
            }

        return self._record(
            calls,
            "escalate_to_human",
            {
                "user_id": user_id,
                "reason": reason,
                "conversation_summary": conversation_summary,
                "priority": priority,
            },
            run,
        )

    def update_customer_memory(self, calls: list[ToolCall], user_id: str, memory_item: dict[str, Any]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            stored = self.memory.update(user_id, memory_item)
            return stored

        return self._record(calls, "update_customer_memory", {"user_id": user_id, "memory_item": memory_item}, run)

    def retrieve_customer_memory(self, calls: list[ToolCall], user_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            validate_user_id(user_id)
            return {"memory": self.memory.retrieve(user_id)}

        return self._record(calls, "retrieve_customer_memory", {"user_id": user_id}, run)
