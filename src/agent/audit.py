from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        tool_name: str,
        user_id: str,
        args: dict[str, Any],
        ok: bool,
        result: dict[str, Any],
    ) -> None:
        safe_args = dict(args)
        if "memory_item" in safe_args:
            safe_args["memory_item"] = "[redacted-for-audit]"
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "user_id": user_id,
            "args": safe_args,
            "ok": ok,
            "result": result,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
