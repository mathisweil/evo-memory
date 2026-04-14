# Divergent Samples Report (extended_test split)
Samples with the largest F1 difference between key condition pairs.

## M1 vs M4/cs1024 on 2WikiMQA
*M4 massively beats M1 on 2WikiMQA*
### M1 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 29 | 1.000 | 0.000 | +1.000 |
| 53 | 1.000 | 0.000 | +1.000 |
| 68 | 1.000 | 0.000 | +1.000 |
| 69 | 1.000 | 0.000 | +1.000 |
| 77 | 1.000 | 0.000 | +1.000 |

### M4/cs1024 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 67 | 0.000 | 1.000 | -1.000 |
| 103 | 0.000 | 1.000 | -1.000 |
| 113 | 0.000 | 1.000 | -1.000 |
| 143 | 0.000 | 1.000 | -1.000 |
| 147 | 0.000 | 1.000 | -1.000 |

**Example: prompt 29 (M1 wins)**

*M1* (F1=1.000, type=correct):
- **Pred**: ` Blue Blood And Red`
- **Gold**: `Blue Blood And Red`

*M4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `The Longshot`
- **Gold**: `Blue Blood And Red`

**Example: prompt 67 (M4/cs1024 wins)**

*M1* (F1=0.000, type=wrong):
- **Pred**: ` Closely Watched Trains`
- **Gold**: `Det Sande Ansigt`

*M4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Det Sande Ansigt`
- **Gold**: `Det Sande Ansigt`

## M1 vs M4/cs1024 on Qasper
*M1 beats M4 on Qasper*
### M1 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 176 | 1.000 | 0.000 | +1.000 |
| 179 | 1.000 | 0.000 | +1.000 |
| 114 | 0.750 | 0.194 | +0.556 |
| 158 | 0.500 | 0.000 | +0.500 |
| 165 | 0.483 | 0.000 | +0.483 |

### M4/cs1024 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 161 | 0.000 | 1.000 | -1.000 |
| 168 | 0.058 | 0.571 | -0.513 |
| 180 | 0.211 | 0.545 | -0.335 |
| 7 | 0.098 | 0.400 | -0.302 |
| 101 | 0.000 | 0.300 | -0.300 |

**Example: prompt 176 (M1 wins)**

*M1* (F1=1.000, type=correct):
- **Pred**: ` 6946`
- **Gold**: `3606 | 6946`

*M4/cs1024* (F1=0.000, type=abstention):
- **Pred**: `Unanswerable`
- **Gold**: `3606 | 6946`

**Example: prompt 161 (M4/cs1024 wins)**

*M1* (F1=0.000, type=hallucination):
- **Pred**: ` They used unigrams, linguistic features, and features derived from a graph structure.`
- **Gold**: `Unanswerable`

*M4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Unanswerable`
- **Gold**: `Unanswerable`

## B0 vs M1 on HotpotQA-E
*Both strong on HotpotQA-E*
### B0 wins (top 5):
| prompt_idx | B0 F1 | M1 F1 | delta |
|---|---|---|---|
| 89 | 1.000 | 0.000 | +1.000 |
| 99 | 1.000 | 0.000 | +1.000 |
| 124 | 1.000 | 0.667 | +0.333 |
| 118 | 0.154 | 0.000 | +0.154 |
| 96 | 0.133 | 0.000 | +0.133 |

### M1 wins (top 5):
| prompt_idx | B0 F1 | M1 F1 | delta |
|---|---|---|---|
| 93 | 0.000 | 1.000 | -1.000 |
| 121 | 0.000 | 1.000 | -1.000 |
| 101 | 0.000 | 0.667 | -0.667 |
| 137 | 0.154 | 0.667 | -0.513 |
| 117 | 0.214 | 0.261 | -0.047 |

**Example: prompt 89 (B0 wins)**

*B0* (F1=1.000, type=correct):
- **Pred**: `Marlborough`
- **Gold**: `Marlborough`

*M1* (F1=0.000, type=wrong):
- **Pred**: ` East Jaffrey Historic District`
- **Gold**: `Marlborough`

**Example: prompt 93 (M1 wins)**

*B0* (F1=0.000, type=wrong):
- **Pred**: `No`
- **Gold**: `yes`

*M1* (F1=1.000, type=correct):
- **Pred**: ` Yes`
- **Gold**: `yes`

## M4/cs1024 vs A4/cs1024 on 2WikiMQA
*M4(NAMM+LoRA) vs A4(LoRA-only)*
### M4/cs1024 wins (top 5):
| prompt_idx | M4/cs1024 F1 | A4/cs1024 F1 | delta |
|---|---|---|---|
| 103 | 1.000 | 0.000 | +1.000 |
| 143 | 1.000 | 0.000 | +1.000 |
| 151 | 1.000 | 0.000 | +1.000 |
| 155 | 1.000 | 0.000 | +1.000 |
| 23 | 0.667 | 0.000 | +0.667 |

### A4/cs1024 wins (top 5):
| prompt_idx | M4/cs1024 F1 | A4/cs1024 F1 | delta |
|---|---|---|---|
| 140 | 0.000 | 1.000 | -1.000 |
| 0 | 0.000 | 0.667 | -0.667 |
| 22 | 0.000 | 0.400 | -0.400 |
| 153 | 0.000 | 0.333 | -0.333 |
| 5 | 0.000 | 0.333 | -0.333 |

**Example: prompt 103 (M4/cs1024 wins)**

*M4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Dhuen Ki Lakeer`
- **Gold**: `Dhuen Ki Lakeer`

*A4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `Bomma Borusa`
- **Gold**: `Dhuen Ki Lakeer`

**Example: prompt 140 (A4/cs1024 wins)**

*M4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `Brooklyn`
- **Gold**: `Poznań`

*A4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Poznań`
- **Gold**: `Poznań`
