# LoRA vs NAMM Attribution (test)

## Budget: 1024 tokens

| Task | Trunc/plain | +LoRA (delta) | +NAMM→M4 (delta) | LoRA% | NAMM% |
|---|---|---|---|---|---|
| Qasper | 29.8 | 26.4 (-3.4) | 29.3 (+2.9) | 693% | -593% |
| 2WikiMQA | 26.5 | 26.5 (+0.0) | 44.2 (+17.7) | 0% | 100% |
| Qasper-E | 14.0 | 27.2 (+13.2) | 26.6 (-0.6) | 105% | -5% |
| HotpotQA-E | 9.4 | 33.9 (+24.5) | 43.5 (+9.6) | 72% | 28% |
| 2WikiMQA-E | 12.5 | 21.4 (+8.9) | 22.8 (+1.4) | 87% | 13% |
| **MEAN** | 18.4 | 27.1 (+8.6) | 33.3 (+6.2) | 58% | 42% |

**LoRA contributes +8.6 F1, NAMM adds +6.2 F1**

NAMM-only (M2 vs Trunc/plain): M2=21.1, Trunc=18.4, NAMM alone adds +2.7

## Budget: 2048 tokens

| Task | Trunc/plain | +LoRA (delta) | +NAMM→M4 (delta) | LoRA% | NAMM% |
|---|---|---|---|---|---|
| Qasper | 24.8 | 31.6 (+6.8) | 39.7 (+8.1) | 45% | 55% |
| 2WikiMQA | 25.0 | 27.6 (+2.6) | 25.0 (-2.6) | 0% | 0% |
| Qasper-E | 12.4 | 30.0 (+17.6) | 30.5 (+0.4) | 98% | 2% |
| HotpotQA-E | 17.3 | 34.0 (+16.7) | 35.5 (+1.6) | 91% | 9% |
| 2WikiMQA-E | 14.3 | 21.4 (+7.1) | 24.6 (+3.2) | 69% | 31% |
| **MEAN** | 18.8 | 28.9 (+10.1) | 31.1 (+2.1) | 83% | 17% |

**LoRA contributes +10.1 F1, NAMM adds +2.1 F1**

NAMM-only (M2 vs Trunc/plain): M2=18.3, Trunc=18.8, NAMM alone adds -0.5

