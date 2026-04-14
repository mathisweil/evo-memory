# D2: Length Scaling Analysis

Systematic comparison of test (4K-6.5K tokens) vs extended_test (up to 8K tokens).

## Overall scaling (mean F1 across tasks)

| Condition | Test F1 | Ext F1 | Delta | %Change |
|---|---|---|---|---|
| A4/2K (LoRA-only) | 34.1 | 26.0 | -8.1 | -23.8% |
| M4/2K (LoRA+NAMM) | 31.1 | 23.4 | -7.7 | -24.8% |
| M4/1K (LoRA+NAMM) | 33.3 | 28.0 | -5.2 | -15.7% |
| B1/1K (rand NAMM) | 12.8 | 8.8 | -4.0 | -31.4% |
| B1/2K (rand NAMM) | 14.4 | 11.7 | -2.6 | -18.4% |
| A4/1K (LoRA-only) | 28.7 | 26.4 | -2.3 | -8.0% |
| Trunc+LoRA/1K | 27.1 | 25.7 | -1.4 | -5.1% |
| B0 (vanilla) | 24.1 | 23.7 | -0.4 | -1.7% |
| M2/1K (rand LoRA+NAMM) | 21.1 | 20.8 | -0.4 | -1.8% |
| M1rec/1K (broken) | 0.0 | 0.0 | +0.0 | +nan% |
| Trunc/1K | 18.4 | 18.8 | +0.3 | +1.9% |
| Trunc+LoRA/2K | 28.9 | 29.3 | +0.4 | +1.5% |
| M2/2K (rand LoRA+NAMM) | 18.3 | 19.3 | +1.0 | +5.4% |
| Trunc/2K | 18.8 | 20.3 | +1.5 | +8.1% |
| M1 (NAMM-only) | 30.3 | 33.4 | +3.2 | +10.4% |

## Per-task deltas (ext - test F1)


### B0 (vanilla)
| Task | Test F1 | Ext F1 | Delta |
|---|---|---|---|
| Qasper | 25.9 | 18.3 | -7.5 |
| 2WikiMQA | 26.5 | 17.9 | -8.7 |
| Qasper-E | 6.1 | 13.1 | +7.1 |
| HotpotQA-E | 44.6 | 45.9 | +1.3 |
| 2WikiMQA-E | 17.5 | 23.3 | +5.8 |

### M1 (NAMM-only)
| Task | Test F1 | Ext F1 | Delta |
|---|---|---|---|
| Qasper | 45.0 | 35.9 | -9.1 |
| 2WikiMQA | 10.0 | 23.6 | +13.6 |
| Qasper-E | 35.6 | 27.8 | -7.8 |
| HotpotQA-E | 30.5 | 47.8 | +17.3 |
| 2WikiMQA-E | 30.2 | 31.9 | +1.8 |

### M4/1K (LoRA+NAMM)
| Task | Test F1 | Ext F1 | Delta |
|---|---|---|---|
| Qasper | 29.3 | 25.4 | -3.9 |
| 2WikiMQA | 44.2 | 27.3 | -17.0 |
| Qasper-E | 26.6 | 23.8 | -2.8 |
| HotpotQA-E | 43.5 | 41.2 | -2.2 |
| 2WikiMQA-E | 22.8 | 22.5 | -0.3 |

### A4/1K (LoRA-only)
| Task | Test F1 | Ext F1 | Delta |
|---|---|---|---|
| Qasper | 46.2 | 30.4 | -15.8 |
| 2WikiMQA | 25.0 | 23.6 | -1.4 |
| Qasper-E | 28.1 | 21.4 | -6.7 |
| HotpotQA-E | 26.7 | 31.6 | +4.9 |
| 2WikiMQA-E | 17.5 | 25.1 | +7.6 |

### Trunc+LoRA/2K
| Task | Test F1 | Ext F1 | Delta |
|---|---|---|---|
| Qasper | 31.6 | 30.6 | -0.9 |
| 2WikiMQA | 27.6 | 18.7 | -8.8 |
| Qasper-E | 30.0 | 31.7 | +1.6 |
| HotpotQA-E | 34.0 | 41.9 | +7.9 |
| 2WikiMQA-E | 21.4 | 23.8 | +2.4 |

## Key findings

**Most robust to longer context** (smallest degradation or improvement):
- M1 (NAMM-only): +3.2
- Trunc/2K: +1.5
- M2/2K (rand LoRA+NAMM): +1.0

**Most fragile at longer context**:
- A4/2K (LoRA-only): -8.1
- M4/2K (LoRA+NAMM): -7.7
- M4/1K (LoRA+NAMM): -5.2

## F1 by length bin (extended_test)


**B0 (vanilla)**:
- short (<5K): F1=18.5% (n=168)
- medium (5-6.5K): F1=33.8% (n=56)

**M1 (NAMM-only)**:
- short (<5K): F1=26.7% (n=168)
- medium (5-6.5K): F1=47.1% (n=56)

**M4/1K (LoRA+NAMM)**:
- short (<5K): F1=24.6% (n=168)
- medium (5-6.5K): F1=33.9% (n=56)

**A4/1K (LoRA-only)**:
- short (<5K): F1=23.4% (n=168)
- medium (5-6.5K): F1=32.2% (n=56)

**Trunc+LoRA/2K**:
- short (<5K): F1=25.9% (n=168)
- medium (5-6.5K): F1=33.1% (n=56)
