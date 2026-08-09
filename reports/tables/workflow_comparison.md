| Method | Accuracy | Compliance recall | Risk recall | Budget adherence | Citation validity | Unsafe action rate | Human-gate | Latency (ms) | Tool calls |
|---|---|---|---|---|---|---|---|---|---|
| rules_baseline | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.0000 | 0.3333 | 0.6000 | 0.28 | 0 |
| rag_only | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.6667 | 0.3333 | 0.6000 | 1.81 | 0 |
| rag_mcp | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.6667 | 0.3333 | 0.6000 | 23.43 | 24 |
| guarded_rag_mcp | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 1.0000 | 0.0000 | 1.0000 | 14.32 | 16 |