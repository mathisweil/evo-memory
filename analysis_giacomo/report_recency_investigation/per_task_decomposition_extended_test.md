# Per-Task Decomposition: Why Trunc/lora ≈ M4 (extended_test)

## 2048-token budget

| Task | Trunc/lora | M4 | M4-Trunc | Verdict |
|---|---|---|---|---|
| Qasper | 30.6 | 23.4 | -7.2 | **Trunc wins** |
| 2WikiMQA | 18.7 | 22.0 | +3.3 | **NAMM helps** |
| Qasper-E | 31.7 | 14.6 | -17.1 | **Trunc wins** |
| HotpotQA-E | 41.9 | 29.6 | -12.3 | **Trunc wins** |
| 2WikiMQA-E | 23.8 | 27.2 | +3.4 | **NAMM helps** |
| **Mean** | **29.3** | **23.4** | **-6.0** | |

## 1024-token budget

| Task | Trunc/lora | M4 | M4-Trunc | Verdict |
|---|---|---|---|---|
| Qasper | 25.5 | 25.4 | -0.1 | ~tied |
| 2WikiMQA | 18.6 | 27.3 | +8.7 | **NAMM helps** |
| Qasper-E | 28.1 | 23.8 | -4.4 | **Trunc wins** |
| HotpotQA-E | 37.3 | 41.2 | +3.9 | **NAMM helps** |
| 2WikiMQA-E | 19.0 | 22.5 | +3.5 | **NAMM helps** |
| **Mean** | **25.7** | **28.0** | **+2.3** | |

