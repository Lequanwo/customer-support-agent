from __future__ import annotations

from pathlib import Path

from agent.memory import CustomerMemoryStore
from agent.security import SecurityError


def test_memory_read_is_user_scoped(data_dir: Path) -> None:
    memory = CustomerMemoryStore(data_dir / "memory_store.json")

    assert memory.retrieve("user_alex")["preferred_contact"] == "email"
    assert "preferred_contact" not in memory.retrieve("user_bri")


def test_safe_memory_write(data_dir: Path) -> None:
    memory = CustomerMemoryStore(data_dir / "memory_store.json")
    memory.update("user_dana", {"key": "preferred_contact", "value": "phone"})

    assert memory.retrieve("user_dana")["preferred_contact"] == "phone"


def test_unsafe_memory_write_blocked(data_dir: Path) -> None:
    memory = CustomerMemoryStore(data_dir / "memory_store.json")

    try:
        memory.update("user_dana", {"key": "preferred_contact", "value": "password is hunter2"})
    except SecurityError as exc:
        assert "Sensitive" in str(exc)
    else:
        raise AssertionError("Expected sensitive memory write to fail")
