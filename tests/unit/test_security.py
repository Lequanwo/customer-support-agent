from __future__ import annotations

from agent.security import contains_sensitive_info, validate_order_id


def test_sensitive_detector_flags_payment_cards() -> None:
    assert contains_sensitive_info("4111 1111 1111 1111")


def test_order_id_validation_rejects_bad_shape() -> None:
    try:
        validate_order_id("../ORD-1001")
    except ValueError as exc:
        assert "Invalid order_id" in str(exc)
    else:
        raise AssertionError("Expected invalid order id to be rejected")
