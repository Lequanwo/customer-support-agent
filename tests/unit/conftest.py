from __future__ import annotations

import shutil
import os
from pathlib import Path

import pytest

from agent import CustomerSupportAgent, MockLLM


ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config) -> None:
    os.environ["MEMORY_BACKEND"] = "json"
    os.environ.pop("CUSTOMER_AGENT_STRICT_LLM", None)


@pytest.fixture(scope="session")
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(ROOT / "data", target)
    (target / "audit_log.jsonl").write_text("", encoding="utf-8")
    return target


@pytest.fixture()
def optimized_agent(data_dir: Path, mock_llm: MockLLM) -> CustomerSupportAgent:
    return CustomerSupportAgent(mode="optimized", data_dir=data_dir, audit_path=data_dir / "audit_log.jsonl", llm=mock_llm)
