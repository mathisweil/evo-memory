# Cross-Condition Error Correlation (extended_test)

## M4/cs2048 vs Trunc/lora_2048

| Task | Both fail | Only M4/cs2048 | Only Trunc/lora_2048 | Both succeed | Jaccard(fail) |
|---|---|---|---|---|---|
| Qasper | 9 | 5 | 3 | 3 | 0.529 |
| 2WikiMQA | 35 | 5 | 5 | 5 | 0.778 |
| Qasper-E | 14 | 12 | 2 | 3 | 0.500 |
| HotpotQA-E | 15 | 5 | 2 | 9 | 0.682 |
| 2WikiMQA-E | 34 | 3 | 11 | 9 | 0.708 |
| **Overall** | **107** | **30** | **23** | **29** | **0.669** |

**M4/cs2048 fails but Trunc/lora_2048 succeeds**: 15 samples
**Trunc/lora_2048 fails but M4/cs2048 succeeds**: 12 samples

### Samples where Trunc/lora_2048 succeeds but M4/cs2048 fails:

- 2WikiMQA[0]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=0.667
- 2WikiMQA[68]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=1.000
- 2WikiMQA[69]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=1.000
- 2WikiMQA-E[101]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=1.000
- 2WikiMQA-E[107]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=1.000

### Samples where M4/cs2048 succeeds but Trunc/lora_2048 fails:

- 2WikiMQA[29]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- 2WikiMQA[53]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- 2WikiMQA[102]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- 2WikiMQA[103]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- 2WikiMQA-E[114]: M4/cs2048 F1=0.667, Trunc/lora_2048 F1=0.000

## M4/cs1024 vs Trunc/lora_1024

| Task | Both fail | Only M4/cs1024 | Only Trunc/lora_1024 | Both succeed | Jaccard(fail) |
|---|---|---|---|---|---|
| Qasper | 7 | 8 | 6 | 2 | 0.333 |
| 2WikiMQA | 31 | 5 | 10 | 5 | 0.674 |
| Qasper-E | 10 | 10 | 10 | 5 | 0.333 |
| HotpotQA-E | 14 | 1 | 5 | 11 | 0.700 |
| 2WikiMQA-E | 35 | 5 | 13 | 6 | 0.660 |
| **Overall** | **97** | **29** | **44** | **29** | **0.571** |

**M4/cs1024 fails but Trunc/lora_1024 succeeds**: 16 samples
**Trunc/lora_1024 fails but M4/cs1024 succeeds**: 23 samples

### Samples where Trunc/lora_1024 succeeds but M4/cs1024 fails:

- 2WikiMQA[3]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- 2WikiMQA[95]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- 2WikiMQA[101]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- 2WikiMQA[153]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=0.667
- 2WikiMQA[154]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000

### Samples where M4/cs1024 succeeds but Trunc/lora_1024 fails:

- 2WikiMQA[23]: M4/cs1024 F1=0.667, Trunc/lora_1024 F1=0.000
- 2WikiMQA[34]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000
- 2WikiMQA[41]: M4/cs1024 F1=0.667, Trunc/lora_1024 F1=0.000
- 2WikiMQA[75]: M4/cs1024 F1=0.600, Trunc/lora_1024 F1=0.000
- 2WikiMQA[102]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000

