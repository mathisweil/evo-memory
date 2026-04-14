# Cross-Condition Error Correlation (test)

## M4/cs2048 vs Trunc/lora_2048

| Task | Both fail | Only M4/cs2048 | Only Trunc/lora_2048 | Both succeed | Jaccard(fail) |
|---|---|---|---|---|---|
| Qasper | 4 | 1 | 2 | 3 | 0.571 |
| 2WikiMQA | 8 | 1 | 0 | 3 | 0.889 |
| Qasper-E | 7 | 2 | 3 | 4 | 0.583 |
| HotpotQA-E | 7 | 0 | 1 | 4 | 0.875 |
| 2WikiMQA-E | 9 | 1 | 2 | 2 | 0.750 |
| **Overall** | **35** | **5** | **8** | **16** | **0.729** |

**M4/cs2048 fails but Trunc/lora_2048 succeeds**: 2 samples
**Trunc/lora_2048 fails but M4/cs2048 succeeds**: 3 samples

### Samples where Trunc/lora_2048 succeeds but M4/cs2048 fails:

- 2WikiMQA-E[107]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=1.000
- Qasper-E[180]: M4/cs2048 F1=0.000, Trunc/lora_2048 F1=0.852

### Samples where M4/cs2048 succeeds but Trunc/lora_2048 fails:

- 2WikiMQA-E[135]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- Qasper[161]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000
- Qasper-E[164]: M4/cs2048 F1=1.000, Trunc/lora_2048 F1=0.000

## M4/cs1024 vs Trunc/lora_1024

| Task | Both fail | Only M4/cs1024 | Only Trunc/lora_1024 | Both succeed | Jaccard(fail) |
|---|---|---|---|---|---|
| Qasper | 4 | 3 | 3 | 1 | 0.400 |
| 2WikiMQA | 5 | 1 | 3 | 2 | 0.556 |
| Qasper-E | 8 | 2 | 2 | 3 | 0.667 |
| HotpotQA-E | 6 | 0 | 2 | 4 | 0.750 |
| 2WikiMQA-E | 8 | 1 | 3 | 2 | 0.667 |
| **Overall** | **31** | **7** | **13** | **12** | **0.608** |

**M4/cs1024 fails but Trunc/lora_1024 succeeds**: 4 samples
**Trunc/lora_1024 fails but M4/cs1024 succeeds**: 7 samples

### Samples where Trunc/lora_1024 succeeds but M4/cs1024 fails:

- 2WikiMQA[154]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- 2WikiMQA-E[107]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- Qasper[179]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=1.000
- Qasper-E[180]: M4/cs1024 F1=0.000, Trunc/lora_1024 F1=0.852

### Samples where M4/cs1024 succeeds but Trunc/lora_1024 fails:

- 2WikiMQA[147]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000
- 2WikiMQA[151]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000
- 2WikiMQA[155]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000
- 2WikiMQA-E[171]: M4/cs1024 F1=0.571, Trunc/lora_1024 F1=0.000
- HotpotQA-E[88]: M4/cs1024 F1=1.000, Trunc/lora_1024 F1=0.000

