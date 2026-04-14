# LoRA vs NAMM Attribution (extended_test)

## Budget: 1024 tokens

| Task | Trunc/plain | +LoRA (delta) | +NAMM→M4 (delta) | LoRA% | NAMM% |
|---|---|---|---|---|---|
| Qasper | 23.4 | 25.5 (+2.1) | 25.4 (-0.1) | 105% | -5% |
| 2WikiMQA | 19.0 | 18.6 (-0.4) | 27.3 (+8.7) | -4% | 104% |
| Qasper-E | 14.5 | 28.1 (+13.6) | 23.8 (-4.4) | 147% | -47% |
| HotpotQA-E | 23.5 | 37.3 (+13.8) | 41.2 (+3.9) | 78% | 22% |
| 2WikiMQA-E | 13.5 | 19.0 (+5.5) | 22.5 (+3.5) | 61% | 39% |
| **MEAN** | 18.8 | 25.7 (+6.9) | 28.0 (+2.3) | 75% | 25% |

**LoRA contributes +6.9 F1, NAMM adds +2.3 F1**

NAMM-only (M2 vs Trunc/plain): M2=20.8, Trunc=18.8, NAMM alone adds +2.0

## Budget: 2048 tokens

| Task | Trunc/plain | +LoRA (delta) | +NAMM→M4 (delta) | LoRA% | NAMM% |
|---|---|---|---|---|---|
| Qasper | 20.0 | 30.6 (+10.7) | 23.4 (-7.2) | 309% | -209% |
| 2WikiMQA | 17.2 | 18.7 (+1.5) | 22.0 (+3.3) | 31% | 69% |
| Qasper-E | 12.7 | 31.7 (+19.0) | 14.6 (-17.1) | 1018% | -918% |
| HotpotQA-E | 31.9 | 41.9 (+9.9) | 29.6 (-12.3) | -422% | 522% |
| 2WikiMQA-E | 19.5 | 23.8 (+4.3) | 27.2 (+3.4) | 56% | 44% |
| **MEAN** | 20.3 | 29.3 (+9.1) | 23.4 (-6.0) | 294% | -194% |

**LoRA contributes +9.1 F1, NAMM adds -6.0 F1**

NAMM-only (M2 vs Trunc/plain): M2=19.3, Trunc=20.3, NAMM alone adds -1.0

