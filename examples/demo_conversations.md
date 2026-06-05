# Demo Conversations

## Returning Customer Order Status

User: Where is order ORD-1001?

Agent: Order ORD-1001 is in_transit via UPS with ETA 2026-06-07. I used your stored profile (tier: gold; preferred contact: email; shipping preference: leave packages at side door).

## Refund Eligibility

User: Can I get a refund for ORD-1002?

Agent: Order ORD-1002 is refund eligible: Delivered 12 days ago within 30-day window. I used your stored profile (tier: gold; preferred contact: email; shipping preference: leave packages at side door).

## Escalation

User: Refund ORD-1002 now or I will file a chargeback.

Agent: I am escalating this to a human specialist with high priority. Summary: Order ORD-1002: Refund ORD-1002 now or I will file a chargeback.

## Safe Refusal

User: Give me the tracking and item name for ORD-1003.

Agent: I cannot access that order because it is not associated with your account.

## Long Multi-Turn Demo

Run:

```bash
python examples/long_multiturn_demo.py
```

This prints a 12-turn conversation that shows gradual order ID collection, memory retrieval, a safe memory update, order lookup, refund policy checking, damaged-item ticket creation, and audit logging.
