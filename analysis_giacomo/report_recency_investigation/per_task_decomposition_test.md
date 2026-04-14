# Per-Task Decomposition: Why Trunc/lora ≈ M4 (test)

## 2048-token budget

| Task | Trunc/lora | M4 | M4-Trunc | Verdict |
|---|---|---|---|---|
| Qasper | 31.6 | 39.7 | +8.1 | **NAMM helps** |
| 2WikiMQA | 27.6 | 25.0 | -2.6 | ~tied |
| Qasper-E | 30.0 | 30.5 | +0.4 | ~tied |
| HotpotQA-E | 34.0 | 35.5 | +1.6 | ~tied |
| 2WikiMQA-E | 21.4 | 24.6 | +3.2 | **NAMM helps** |
| **Mean** | **28.9** | **31.1** | **+2.1** | |

## 1024-token budget

| Task | Trunc/lora | M4 | M4-Trunc | Verdict |
|---|---|---|---|---|
| Qasper | 26.4 | 29.3 | +2.9 | ~tied |
| 2WikiMQA | 26.5 | 44.2 | +17.7 | **NAMM helps** |
| Qasper-E | 27.2 | 26.6 | -0.6 | ~tied |
| HotpotQA-E | 33.9 | 43.5 | +9.6 | **NAMM helps** |
| 2WikiMQA-E | 21.4 | 22.8 | +1.4 | ~tied |
| **Mean** | **27.1** | **33.3** | **+6.2** | |

