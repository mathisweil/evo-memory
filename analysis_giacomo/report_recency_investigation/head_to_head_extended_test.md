# Head-to-Head: Trunc/lora vs M4 (extended_test)

## Trunc/lora_2048 vs M4/cs2048 (Same 2K budget)

| Task | n | Trunc/lora_2048 wins | M4/cs2048 wins | Ties | Trunc/lora_2048 F1 | M4/cs2048 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 32 | 15 | 6 | 11 | 30.6 | 23.4 | +7.2 |
| 2WikiMQA | 54 | 6 | 7 | 41 | 18.7 | 22.0 | -3.3 |
| Qasper-E | 46 | 25 | 5 | 16 | 31.7 | 14.5 | +17.1 |
| HotpotQA-E | 31 | 5 | 2 | 24 | 41.9 | 29.6 | +12.3 |
| 2WikiMQA-E | 61 | 8 | 11 | 42 | 23.8 | 27.2 | -3.4 |

## Trunc/lora_1024 vs M4/cs1024 (Same 1K budget)

| Task | n | Trunc/lora_1024 wins | M4/cs1024 wins | Ties | Trunc/lora_1024 F1 | M4/cs1024 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 32 | 12 | 11 | 9 | 25.5 | 25.4 | +0.1 |
| 2WikiMQA | 54 | 5 | 13 | 36 | 18.6 | 27.3 | -8.7 |
| Qasper-E | 46 | 19 | 15 | 12 | 28.1 | 23.7 | +4.4 |
| HotpotQA-E | 31 | 2 | 5 | 24 | 37.3 | 41.2 | -3.9 |
| 2WikiMQA-E | 61 | 7 | 14 | 40 | 19.0 | 22.5 | -3.5 |

## Trunc/lora_2048 vs M4/cs1024 (Trunc has 2x budget)

| Task | n | Trunc/lora_2048 wins | M4/cs1024 wins | Ties | Trunc/lora_2048 F1 | M4/cs1024 F1 | Delta |
|---|---|---|---|---|---|---|---|
| Qasper | 32 | 13 | 8 | 11 | 30.6 | 25.4 | +5.3 |
| 2WikiMQA | 54 | 4 | 11 | 39 | 18.7 | 27.3 | -8.5 |
| Qasper-E | 46 | 22 | 12 | 12 | 31.7 | 23.7 | +8.0 |
| HotpotQA-E | 31 | 4 | 5 | 22 | 41.9 | 41.2 | +0.6 |
| 2WikiMQA-E | 61 | 9 | 11 | 41 | 23.8 | 22.5 | +1.3 |

