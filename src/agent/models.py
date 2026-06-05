from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any]


@dataclass
class AgentResult:
    user_id: str
    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    should_escalate: bool = False
    memory_reads: int = 0
    memory_writes: int = 0
    security_violations: int = 0
    escalations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]
