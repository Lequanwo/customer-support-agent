from __future__ import annotations

import re
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .memory import describe_memory
from .models import AgentResult, ToolCall
from .policy import EscalationDecision, decide_escalation
from .security import contains_sensitive_info, explicit_human_request, high_risk_text
from .tools import SupportTools


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        if prompt.startswith("TASK: understand_query"):
            payload = self._payload(prompt)
            transcript = payload.get("transcript", "")
            latest = payload.get("latest", "")
            understanding = {
                "order_id": self._extract_order_id(transcript),
                "wants_refund": self._has_any(transcript, ["refund", "return", "money back"]),
                "wants_status": self._has_any(
                    transcript,
                    ["where is", "status", "track", "shipping", "delayed", "late", "show me"],
                ),
                "damaged": self._has_any(transcript, ["damaged", "broken", "cracked"]),
                "memory_item": self._extract_memory_update(latest),
            }
            understanding["needs_order_id"] = (
                self._has_any(transcript, ["order", "refund", "return", "shipping", "damaged", "track", "where is"])
                and not understanding["order_id"]
            )
            return json.dumps(understanding, sort_keys=True)
        if prompt.startswith("TASK: create_action_plan"):
            payload = self._payload(prompt)
            understanding = payload.get("understanding", {})
            order_id = understanding.get("order_id")
            plan = {
                "needs_clarification": bool(understanding.get("needs_order_id")),
                "should_lookup_order": bool(
                    order_id
                    and (
                        understanding.get("wants_status")
                        or understanding.get("wants_refund")
                        or understanding.get("damaged")
                    )
                ),
                "should_check_refund_policy": bool(order_id and understanding.get("wants_refund")),
                "should_create_damage_ticket": bool(order_id and understanding.get("damaged")),
                "should_update_memory": bool(understanding.get("memory_item")),
                "rationale": "Use customer-scoped tools only after validation; let governance decide escalation.",
            }
            return json.dumps(plan, sort_keys=True)
        if prompt.startswith("TASK: decide_escalation"):
            payload = self._payload(prompt)
            decision = decide_escalation(
                payload.get("transcript", ""),
                repeated_issue=bool(payload.get("repeated_issue")),
                ambiguous_policy=bool(payload.get("ambiguous_policy")),
                low_confidence=bool(payload.get("low_confidence")),
            )
            return json.dumps(
                {
                    "should_escalate": decision.should_escalate,
                    "priority": decision.priority,
                    "reason": decision.reason,
                },
                sort_keys=True,
            )
        if prompt.startswith("TASK: summarize_conversation"):
            payload = self._payload(prompt)
            return json.dumps(
                {
                    "summary": self._fallback_summary(
                        payload.get("transcript", ""),
                        payload.get("order_id"),
                    )
                },
                sort_keys=True,
            )
        if prompt.startswith("TASK: generate_response"):
            payload = self._payload(prompt)
            return json.dumps({"response": payload.get("draft_response", "")}, sort_keys=True)
        return prompt

    @staticmethod
    def _payload(prompt: str) -> dict:
        marker = "PAYLOAD_JSON:"
        if marker not in prompt:
            return {}
        try:
            return json.loads(prompt.split(marker, 1)[1])
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        matches = re.findall(r"\bORD-\d{4}\b", text, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @staticmethod
    def _has_any(text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

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


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        _load_local_env()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        self.base_url = os.getenv("OPENAI_API_BASE_URL", base_url)

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic customer support planning component. "
                        "Return only a valid JSON object. Do not include markdown, prose, or code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
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


def make_llm(kind: str = "mock", *, model: str | None = None) -> BaseLLM:
    if kind == "mock":
        return MockLLM()
    if kind == "openai":
        return OpenAICompatibleLLM(model=model)
    raise ValueError("LLM kind must be 'mock' or 'openai'.")


def _load_local_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
        understanding = self._llm_understand(user_id=user_id, transcript=transcript, latest=latest)

        if self.mode == "optimized":
            mem_result = self.tools.retrieve_customer_memory(calls, user_id)
            memory = mem_result.get("memory", {})
            memory_reads += 1

        plan = self._llm_plan(user_id=user_id, transcript=transcript, understanding=understanding, memory=memory)
        unsafe_memory_request = self._extract_memory_update(latest)
        plan["should_update_memory"] = bool(unsafe_memory_request)
        if unsafe_memory_request and contains_sensitive_info(unsafe_memory_request.get("value", "")):
            result = self.tools.update_customer_memory(calls, user_id, unsafe_memory_request)
            security_violations += int(not result.get("stored"))
            draft = "I cannot store sensitive information such as passwords, payment card numbers, SSNs, or medical/legal details."
            return AgentResult(
                user_id=user_id,
                response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                tool_calls=calls,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                security_violations=security_violations + self._security_count(calls),
            )

        order_id = understanding.get("order_id")
        wants_refund = bool(understanding.get("wants_refund"))
        wants_status = bool(understanding.get("wants_status"))
        damaged = bool(understanding.get("damaged"))
        conversation_summary = self._llm_summarize_conversation(transcript=transcript, order_id=order_id)

        bypass_clarification = explicit_human_request(transcript) or high_risk_text(transcript)
        if plan.get("needs_clarification") and not bypass_clarification:
            if self._has_any(latest, ["do not have", "don't have", "dont have", "not have", "can't find", "cannot find"]):
                draft = (
                    "No problem. I can wait while you find it; the order ID is the safest way for me "
                    "to look up the correct order without accessing the wrong account."
                )
            else:
                draft = "I can help with that. Please share the order ID so I can check the right order."
            return AgentResult(
                user_id=user_id,
                response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                tool_calls=calls,
                memory_reads=memory_reads,
                security_violations=self._security_count(calls),
            )

        repeated_issue = bool(memory.get("unresolved_tickets")) and self._has_any(transcript, ["again", "still", "unresolved", "third time"])
        escalation = self._llm_decide_escalation(transcript=transcript, repeated_issue=repeated_issue)

        if order_id and plan.get("should_lookup_order"):
            order = self.tools.lookup_order(calls, order_id, user_id)
            if order.get("security_violation"):
                draft = "I cannot access that order because it is not associated with your account."
                return AgentResult(
                    user_id=user_id,
                    response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )
            if order.get("error"):
                ticket = self.tools.create_support_ticket(calls, user_id, f"Order lookup failed for {order_id}: {conversation_summary}", "normal")
                draft = f"I could not find {order_id}. I created support ticket {ticket.get('ticket_id')} so the team can retry the lookup."
                return AgentResult(
                    user_id=user_id,
                    response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )

        refund_result = None
        if order_id and plan.get("should_check_refund_policy"):
            refund_result = self.tools.check_refund_policy(calls, order_id, user_id)
            if refund_result.get("security_violation"):
                draft = "I cannot check refund eligibility for an order outside your account."
                return AgentResult(
                    user_id=user_id,
                    response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                    tool_calls=calls,
                    memory_reads=memory_reads,
                    security_violations=self._security_count(calls),
                )
            escalation = self._llm_decide_escalation(
                transcript=transcript,
                repeated_issue=repeated_issue,
                ambiguous_policy=bool(refund_result.get("ambiguous")),
            )

        if plan.get("should_create_damage_ticket") and not escalation.should_escalate:
            self.tools.create_support_ticket(calls, user_id, conversation_summary, "normal")

        if plan.get("should_update_memory") and unsafe_memory_request and self.mode == "optimized":
            stored = self.tools.update_customer_memory(calls, user_id, unsafe_memory_request)
            if stored.get("stored"):
                memory_writes += 1
                saved_memory = True

        if escalation.should_escalate:
            esc = self.tools.escalate_to_human(calls, user_id, escalation.reason, conversation_summary, escalation.priority)
            escalations.append(esc)
            draft = f"I am escalating this to a human specialist with {escalation.priority} priority. Summary: {conversation_summary}"
            return AgentResult(
                user_id=user_id,
                response=self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan),
                tool_calls=calls,
                should_escalate=True,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                security_violations=self._security_count(calls),
                escalations=escalations,
            )

        draft = self._compose_response(
            transcript=transcript,
            saved_memory=saved_memory,
            order_id=order_id,
            wants_status=wants_status,
            wants_refund=wants_refund,
            damaged=damaged,
            refund_result=refund_result,
            calls=calls,
            memory=memory,
            conversation_summary=conversation_summary,
        )
        response = self._llm_generate_response(draft, calls=calls, memory=memory, plan=plan)
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
        conversation_summary: str,
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
        if "summarize" in transcript.lower():
            parts.append(f"Conversation summary: {conversation_summary}")
            parts.append(f"Tools used: {', '.join(call.name for call in calls)}.")
        if not parts:
            parts.append("I can help with order status, refund eligibility, damaged items, or human escalation.")
        return " ".join(parts)

    def _llm_understand(self, *, user_id: str, transcript: str, latest: str) -> dict:
        fallback = {
            "order_id": self._extract_order_id(transcript),
            "wants_refund": self._has_any(transcript, ["refund", "return", "money back"]),
            "wants_status": self._has_any(
                transcript,
                ["where is", "status", "track", "shipping", "delayed", "late", "show me"],
            ),
            "damaged": self._has_any(transcript, ["damaged", "broken", "cracked"]),
            "memory_item": self._extract_memory_update(latest),
        }
        fallback["needs_order_id"] = self._needs_order_id(transcript)
        payload = {"user_id": user_id, "transcript": transcript, "latest": latest}
        prompt = (
            "TASK: understand_query\n"
            "Return JSON with keys: order_id, wants_refund, wants_status, damaged, memory_item, needs_order_id.\n"
            "memory_item must be null or an object with keys key and value.\n"
            "PAYLOAD_JSON:" + json.dumps(payload, sort_keys=True)
        )
        result = self._llm_json(prompt, fallback)
        order_id = result.get("order_id") or fallback["order_id"]
        memory_item = result.get("memory_item")
        if not isinstance(memory_item, dict):
            memory_item = fallback["memory_item"]
        return {
            "order_id": order_id,
            "wants_refund": bool(result.get("wants_refund")) or bool(fallback["wants_refund"]),
            "wants_status": bool(result.get("wants_status")) or bool(fallback["wants_status"]),
            "damaged": bool(result.get("damaged")) or bool(fallback["damaged"]),
            "memory_item": memory_item,
            "needs_order_id": bool(result.get("needs_order_id")) if not order_id else False,
        }

    def _llm_plan(self, *, user_id: str, transcript: str, understanding: dict, memory: dict) -> dict:
        order_id = understanding.get("order_id")
        fallback = {
            "needs_clarification": bool(understanding.get("needs_order_id")),
            "should_lookup_order": bool(
                order_id
                and (
                    understanding.get("wants_status")
                    or understanding.get("wants_refund")
                    or understanding.get("damaged")
                )
            ),
            "should_check_refund_policy": bool(order_id and understanding.get("wants_refund")),
            "should_create_damage_ticket": bool(order_id and understanding.get("damaged")),
            "should_update_memory": bool(understanding.get("memory_item")),
            "rationale": "Fallback deterministic plan.",
        }
        payload = {
            "user_id": user_id,
            "transcript": transcript,
            "understanding": understanding,
            "memory_summary": describe_memory(memory),
        }
        prompt = (
            "TASK: create_action_plan\n"
            "Return JSON with keys: needs_clarification, should_lookup_order, should_check_refund_policy, "
            "should_create_damage_ticket, should_update_memory, rationale.\n"
            "All should_* and needs_clarification values must be booleans.\n"
            "PAYLOAD_JSON:" + json.dumps(payload, sort_keys=True)
        )
        result = self._llm_json(prompt, fallback)
        needs_clarification = bool(result.get("needs_clarification")) or fallback["needs_clarification"]
        if order_id or explicit_human_request(transcript) or high_risk_text(transcript):
            needs_clarification = False
        return {
            "needs_clarification": needs_clarification,
            "should_lookup_order": bool(result.get("should_lookup_order")) or fallback["should_lookup_order"],
            "should_check_refund_policy": bool(result.get("should_check_refund_policy"))
            or fallback["should_check_refund_policy"],
            "should_create_damage_ticket": bool(result.get("should_create_damage_ticket"))
            or fallback["should_create_damage_ticket"],
            "should_update_memory": bool(result.get("should_update_memory")) or fallback["should_update_memory"],
            "rationale": str(result.get("rationale") or fallback["rationale"]),
        }

    def _llm_decide_escalation(
        self,
        *,
        transcript: str,
        repeated_issue: bool = False,
        ambiguous_policy: bool = False,
        low_confidence: bool = False,
    ) -> EscalationDecision:
        fallback = decide_escalation(
            transcript,
            repeated_issue=repeated_issue,
            ambiguous_policy=ambiguous_policy,
            low_confidence=low_confidence,
        )
        payload = {
            "transcript": transcript,
            "repeated_issue": repeated_issue,
            "ambiguous_policy": ambiguous_policy,
            "low_confidence": low_confidence,
        }
        prompt = (
            "TASK: decide_escalation\n"
            "Return JSON with keys: should_escalate, priority, reason.\n"
            "priority must be one of low, normal, high, urgent.\n"
            "Escalate for explicit human requests, chargeback/legal/safety risk, repeated unresolved issues, "
            "ambiguous policy, or low confidence.\n"
            "PAYLOAD_JSON:" + json.dumps(payload, sort_keys=True)
        )
        result = self._llm_json(prompt, {})
        priority = result.get("priority")
        reason = result.get("reason")
        should_escalate = result.get("should_escalate")
        if priority not in {"low", "normal", "high", "urgent"}:
            return fallback
        if not isinstance(reason, str) or not reason:
            return fallback
        if not isinstance(should_escalate, bool):
            return fallback
        if fallback.should_escalate and not should_escalate:
            return fallback
        return EscalationDecision(should_escalate=should_escalate, priority=priority, reason=reason)

    def _llm_summarize_conversation(self, *, transcript: str, order_id: str | None) -> str:
        fallback = self._fallback_summary(transcript, order_id)
        payload = {"transcript": transcript, "order_id": order_id}
        prompt = (
            "TASK: summarize_conversation\n"
            "Return JSON with key summary. Keep the summary concise, factual, and safe for a support handoff.\n"
            "Do not include sensitive information.\n"
            "PAYLOAD_JSON:" + json.dumps(payload, sort_keys=True)
        )
        result = self._llm_json(prompt, {"summary": fallback})
        summary = result.get("summary")
        return summary[:450] if isinstance(summary, str) and summary else fallback

    def _llm_generate_response(
        self,
        draft_response: str,
        *,
        calls: list[ToolCall],
        memory: dict,
        plan: dict,
    ) -> str:
        payload = {
            "draft_response": draft_response,
            "tool_names": [call.name for call in calls],
            "memory_summary": describe_memory(memory),
            "plan": plan,
        }
        prompt = (
            "TASK: generate_response\n"
            "Return JSON with key response. Keep the customer-facing response concise and do not add facts "
            "not present in the draft or tool results.\n"
            "PAYLOAD_JSON:" + json.dumps(payload, sort_keys=True)
        )
        result = self._llm_json(prompt, {"response": draft_response})
        response = result.get("response")
        if not isinstance(response, str) or not response:
            return draft_response
        for required_text in ["in_transit", "could not find", "Please share the order ID", "Order ORD-"]:
            if required_text in draft_response and required_text not in response:
                return draft_response
        if "Conversation summary:" in draft_response and "Conversation summary:" not in response:
            return draft_response
        if "Tools used:" in draft_response and "Tools used:" not in response:
            return draft_response
        return response

    def _llm_json(self, prompt: str, fallback: dict) -> dict:
        try:
            raw = self.llm.complete(prompt)
            parsed = json.loads(raw)
        except Exception:
            if os.getenv("CUSTOMER_AGENT_STRICT_LLM") == "1":
                raise
            return fallback
        if not isinstance(parsed, dict):
            if os.getenv("CUSTOMER_AGENT_STRICT_LLM") == "1":
                raise ValueError("LLM response was not a JSON object.")
            return fallback
        return parsed

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
    def _fallback_summary(text: str, order_id: str | None) -> str:
        lowered = text.lower()
        orders = re.findall(r"\bORD-\d{4}\b", text, re.IGNORECASE)
        active_order = order_id or (orders[-1].upper() if orders else None)
        facts: list[str] = []
        if active_order:
            facts.append(f"Order {active_order}")
        if "preferred contact" in lowered:
            facts.append("customer updated preferred contact information")
        if any(word in lowered for word in ["where is", "status", "track", "shipping", "delayed"]):
            facts.append("customer asked about order status or shipping")
        if any(word in lowered for word in ["refund", "return", "money back"]):
            facts.append("customer asked about refund eligibility")
        if any(word in lowered for word in ["damaged", "broken", "cracked"]):
            facts.append("customer reported a damaged item")
        if any(word in lowered for word in ["human", "manager", "supervisor", "representative", "agent"]):
            facts.append("customer requested human help")
        if any(word in lowered for word in ["frustrated", "furious", "unacceptable", "chargeback", "sue", "lawyer"]):
            facts.append("customer sentiment or risk language was noted")
        if not facts:
            clean = " ".join(text.split())
            return clean[:300]
        return "; ".join(facts) + "."

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
