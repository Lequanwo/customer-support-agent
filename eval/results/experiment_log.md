| Round | Change | Metric delta | Conclusion |
|---|---|---|---|
| 0 | Baseline agent with simple rules | Overall 0.7122, memory 0.0000, escalation 0.9600, tool routing 0.5133 | Establishes benchmark and shows the cost of omitting cross-session memory |
| 1 | Add cross-session memory retrieval | Memory retention improves from 0.0000 to 1.0000 | Useful for returning customers and tier/preference-aware responses |
| 2 | Add stricter escalation policy | Escalation accuracy improves from 0.9600 to 1.0000 | Better safety for repeated unresolved, legal, chargeback, and ambiguous policy cases |
| 3 | Add unsafe memory filtering and permission checks | Security violations remain 0 while optimized pass rate reaches 1.0000 | Safer behavior without leaking unauthorized or sensitive facts |
