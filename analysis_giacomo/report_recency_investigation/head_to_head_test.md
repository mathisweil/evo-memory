# Head-to-Head: Trunc/lora vs M4 (test)

## Trunc/lora_2048 vs M4/cs2048 (Same 2K budget)

| Task | n | Trunc/lora_2048 wins | M4/cs2048 wins | Ties | Trunc/lora_2048 F1 | M4/cs2048 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 14 | 2 | 3 | 9 | 31.6 | 39.7 | -8.1 |
| 2WikiMQA | 12 | 1 | 0 | 11 | 27.6 | 25.0 | +2.6 |
| Qasper-E | 18 | 4 | 5 | 9 | 30.0 | 30.5 | -0.4 |
| HotpotQA-E | 12 | 0 | 1 | 11 | 34.0 | 35.5 | -1.6 |
| 2WikiMQA-E | 14 | 1 | 2 | 11 | 21.4 | 24.6 | -3.2 |

## Trunc/lora_1024 vs M4/cs1024 (Same 1K budget)

| Task | n | Trunc/lora_1024 wins | M4/cs1024 wins | Ties | Trunc/lora_1024 F1 | M4/cs1024 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 14 | 5 | 4 | 5 | 26.4 | 29.3 | -2.9 |
| 2WikiMQA | 12 | 1 | 4 | 7 | 26.5 | 44.2 | -17.7 |
| Qasper-E | 18 | 5 | 3 | 10 | 27.2 | 26.6 | +0.6 |
| HotpotQA-E | 12 | 0 | 2 | 10 | 33.9 | 43.5 | -9.6 |
| 2WikiMQA-E | 14 | 1 | 3 | 10 | 21.4 | 22.8 | -1.4 |

## Trunc/lora_2048 vs M4/cs1024 (Trunc has 2x budget)

| Task | n | Trunc/lora_2048 wins | M4/cs1024 wins | Ties | Trunc/lora_2048 F1 | M4/cs1024 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 14 | 5 | 3 | 6 | 31.6 | 29.3 | +2.3 |
| 2WikiMQA | 12 | 0 | 2 | 10 | 27.6 | 44.2 | -16.7 |
| Qasper-E | 18 | 6 | 3 | 9 | 30.0 | 26.6 | +3.5 |
| HotpotQA-E | 12 | 0 | 2 | 10 | 34.0 | 43.5 | -9.5 |
| 2WikiMQA-E | 14 | 1 | 3 | 10 | 21.4 | 22.8 | -1.4 |

