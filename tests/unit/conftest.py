from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent import CustomerSupportAgent


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(ROOT / "data", target)
    (target / "audit_log.jsonl").write_text("", encoding="utf-8")
    return target


@pytest.fixture()
def optimized_agent(data_dir: Path) -> CustomerSupportAgent:
    return CustomerSupportAgent(mode="optimized", data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl")
