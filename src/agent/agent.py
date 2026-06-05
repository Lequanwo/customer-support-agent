from __future__ import annotations

import re
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .memory import describe_memory
from .models import AgentResult, ToolCall
from .policy import decide_escalation
from .security import contains_sensitive_info
from .tools import SupportTools


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return prompt


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class CustomerSupportAgent:
    def __init__(
        self,
        *,
        mode: str = "optimized",
        data_dir: str | Path = "data",
        audit_path: str | Path = "data/audit_log.jsonl",
        llm: BaseLLM | None = None,
    ):
        if mode not in {"baseline", "optimized"}:
            raise ValueError("mode must be 'baseline' or 'optimized'")
        self.mode = mode
        self.tools = SupportTools(data_dir=data_dir, audit_path=audit_path)
        self.llm = llm or MockLLM()

    def handle(self, user_id: str, input_text: str | list[str]) -> AgentResult:
        turns = [input_text] if isinstance(input_text, str) else list(input_text)
        transcript = "\n".join(turns)
        latest = turns[-1] if turns else ""
        calls: list[ToolCall] = []
        memory: dict = {}
        memory_reads = 0
        memory_writes = 0
        saved_memory = False
        security_violations = 0
        escalations: list[dict] = []

        if self.mode == "optimized":
            mem_result = self.tools.retrieve_customer_memory(calls, user_id)
            memory = mem_result.get("memory", {})
            memory_reads += 1

        unsafe_memory_request = self._extract_memory_update(latest)
        if unsafe_memory_request and contains_sensitive_info(unsafe_memory_request.get("value", "")):
            result = self.tools.update_customer_memory(calls, user_id, unsafe_memory_request)
            security_violations += int(not result.get("stored"))
            return AgentResult(
                user_id=user_id,
                response="I cannot store sensitive information such as passwords, payment card numbers, SSNs, or medical/legal details.",
                tool_calls=calls,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                security_violations=security_violations + self._security_count(calls),
            )

        order_id = self._extract_order_id(transcript)
        wants_refund = self._has_any(transcript, ["refund", "return", "money back"])
        wants_status = self._has_any(transcript, ["where is", "status", "track", "shipping", "delayed", "late", "show me"])
        damaged = self._has_any(transcript, ["damaged", "broken", "cracked"])

        if self._needs_order_id(transcript) and not order_id:
            return AgentResult(
                user_id=user_id,
                response="I can help with that. Please share the order ID so I can check the right order.",
                tool_calls=calls,
                memory_reads=memory_reads,
                security_violations=self._security_count(calls),
            )

        repeated_issue = bool(memory.get("unresolved_tickets")) and self._has_any(transcript, ["again", "still", "unresolved", "third time"])
        escalation = decide_escalation(transcript, repeated_issue=repeated_issue)

        if order_id and (wants_status or wants_refund or damaged):
            order = self.tools.lookup_order(calls, order_id, user_id)
            if order.get("security_violation"):
                return AgentResult(
                    user_id=user_id,
                    response="I cannot access that order because it is not associated with your account.",
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )
            if order.get("error"):
                ticket = self.tools.create_support_ticket(calls, user_id, f"Order lookup failed for {order_id}: {order['error']}", "normal")
                return AgentResult(
                    user_id=user_id,
                    response=f"I could not find {order_id}. I created support ticket {ticket.get('ticket_id')} so the team can retry the lookup.",
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )

        refund_result = None
        if order_id and wants_refund:
            refund_result = self.tools.check_refund_policy(calls, order_id, user_id)
            if refund_result.get("security_violation"):
                return AgentResult(
                    user_id=user_id,
                    response="I cannot check refund eligibility for an order outside your account.",
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )
            escalation = decide_escalation(transcript, repeated_issue=repeated_issue, ambiguous_policy=bool(refund_result.get("ambiguous")))

        if damaged and not escalation.should_escalate:
            self.tools.create_support_ticket(calls, user_id, self._summary(transcript, order_id), "normal")

        if unsafe_memory_request and self.mode == "optimized":
            stored = self.tools.update_customer_memory(calls, user_id, unsafe_memory_request)
            if stored.get("stored"):
                memory_writes += 1
                saved_memory = True

        if escalation.should_escalate:
            esc = self.tools.escalate_to_human(calls, user_id, escalation.reason, self._summary(transcript, order_id), escalation.priority)
            escalations.append(esc)
            return AgentResult(
                user_id=user_id,
                response=f"I am escalating this to a human specialist with {escalation.priority} priority. Summary: {self._summary(transcript, order_id)}",
                tool_calls=calls,
                should_escalate=True,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                security_violations=self._security_count(calls),
                escalations=escalations,
            )

        response = self._compose_response(
            transcript=transcript,
            saved_memory=saved_memory,
            order_id=order_id,
            wants_status=wants_status,
            wants_refund=wants_refund,
            damaged=damaged,
            refund_result=refund_result,
            calls=calls,
            memory=memory,
        )
        return AgentResult(
            user_id=user_id,
            response=response,
            tool_calls=calls,
            should_escalate=False,
            memory_reads=memory_reads,
            memory_writes=memory_writes,
            security_violations=self._security_count(calls),
            escalations=escalations,
        )

    def _compose_response(
        self,
        *,
        transcript: str,
        saved_memory: bool,
        order_id: str | None,
        wants_status: bool,
        wants_refund: bool,
        damaged: bool,
        refund_result: dict | None,
        calls: list[ToolCall],
        memory: dict,
    ) -> str:
        order_call = next((call.result for call in calls if call.name == "lookup_order" and call.ok), None)
        parts: list[str] = []
        if order_call and wants_status:
            parts.append(
                f"Order {order_id} is {order_call.get('status')} via {order_call.get('carrier')} with ETA {order_call.get('eta')}."
            )
        if refund_result:
            if refund_result.get("eligible"):
                parts.append(f"Order {order_id} is refund eligible: {refund_result['reason']}")
            elif refund_result.get("ambiguous"):
                parts.append(f"Order {order_id} needs human refund review: {refund_result['reason']}")
            else:
                parts.append(f"Order {order_id} is not automatically refund eligible: {refund_result['reason']}")
        if damaged:
            parts.append("I created a support ticket for the damaged item and included the order context.")
        if self.mode == "optimized" and memory:
            parts.append(f"I used your stored profile ({describe_memory(memory)}).")
        if saved_memory:
            parts.append("I saved that support preference for future sessions.")
        if not parts:
            parts.append("I can help with order status, refund eligibility, damaged items, or human escalation.")
        return " ".join(parts)

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        matches = re.findall(r"\bORD-\d{4}\b", text, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @staticmethod
    def _has_any(text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

    def _needs_order_id(self, text: str) -> bool:
        return self._has_any(text, ["order", "refund", "return", "shipping", "damaged", "track", "where is"]) and not self._extract_order_id(text)

    @staticmethod
    def _summary(text: str, order_id: str | None) -> str:
        clean = " ".join(text.split())
        prefix = f"Order {order_id}: " if order_id else ""
        return (prefix + clean)[:450]

    @staticmethod
    def _security_count(calls: list[ToolCall]) -> int:
        return sum(1 for call in calls if call.result.get("security_violation"))

    @staticmethod
    def _extract_memory_update(text: str) -> dict | None:
        lowered = text.lower()
        contact = re.search(r"prefer(?:red)? contact (?:method )?(?:is|=)?\s*(email|phone|sms|text)", lowered)
        if contact:
            value = "sms" if contact.group(1) == "text" else contact.group(1)
            return {"key": "preferred_contact", "value": value}
        if "remember" in lowered and "password" in lowered:
            return {"key": "preferred_contact", "value": text}
        if "refund preference" in lowered and "store credit" in lowered:
            return {"key": "refund_preference", "value": "store credit"}
        if "leave packages" in lowered:
            return {"key": "shipping_preference", "value": "leave packages at side door"}
        return None
