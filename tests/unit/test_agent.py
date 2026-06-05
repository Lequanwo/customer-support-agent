from __future__ import annotations


def test_tool_routing_for_order_status(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is order ORD-1001?")

    assert result.tool_names == ["retrieve_customer_memory", "lookup_order"]
    assert "in_transit" in result.response


def test_tool_error_creates_retry_ticket(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is order ORD-9999?")

    assert "lookup_order" in result.tool_names
    assert "create_support_ticket" in result.tool_names
    assert "could not find ORD-9999" in result.response


def test_missing_order_id_clarification(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", "Where is my order?")

    assert "Please share the order ID" in result.response
    assert "lookup_order" not in result.tool_names


def test_multi_turn_missing_order_id_recovery(optimized_agent) -> None:
    result = optimized_agent.handle("user_alex", ["Where is my order?", "It is ORD-1001."])

    assert "lookup_order" in result.tool_names
    assert "Order ORD-1001" in result.response
