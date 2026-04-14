# Divergent Samples Report (test split)
Samples with the largest F1 difference between key condition pairs.

## M1 vs M4/cs1024 on 2WikiMQA
*M4 massively beats M1 on 2WikiMQA*
### M1 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 138 | 0.000 | 0.000 | +0.000 |
| 139 | 0.000 | 0.000 | +0.000 |
| 140 | 0.000 | 0.000 | +0.000 |
| 142 | 0.000 | 0.000 | +0.000 |
| 145 | 1.000 | 1.000 | +0.000 |

### M4/cs1024 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 143 | 0.000 | 1.000 | -1.000 |
| 147 | 0.000 | 1.000 | -1.000 |
| 151 | 0.000 | 1.000 | -1.000 |
| 155 | 0.000 | 1.000 | -1.000 |
| 152 | 0.200 | 0.308 | -0.108 |

**Example: prompt 138 (M1 wins)**

*M1* (F1=0.000, type=wrong):
- **Pred**: ` Ajman`
- **Gold**: `Abu Dhabi`

*M4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `Ajman`
- **Gold**: `Abu Dhabi`

**Example: prompt 143 (M4/cs1024 wins)**

*M1* (F1=0.000, type=wrong):
- **Pred**: ` The Third Kiss`
- **Gold**: `Forbidden Daughters`

*M4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Forbidden Daughters`
- **Gold**: `Forbidden Daughters`

## M1 vs M4/cs1024 on Qasper
*M1 beats M4 on Qasper*
### M1 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 176 | 1.000 | 0.000 | +1.000 |
| 179 | 1.000 | 0.000 | +1.000 |
| 158 | 0.500 | 0.000 | +0.500 |
| 165 | 0.483 | 0.000 | +0.483 |
| 173 | 0.737 | 0.308 | +0.429 |

### M4/cs1024 wins (top 5):
| prompt_idx | M1 F1 | M4/cs1024 F1 | delta |
|---|---|---|---|
| 161 | 0.000 | 1.000 | -1.000 |
| 180 | 0.211 | 1.000 | -0.789 |
| 166 | 0.063 | 0.261 | -0.197 |
| 159 | 0.000 | 0.000 | +0.000 |
| 175 | 0.000 | 0.000 | +0.000 |

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
| 124 | 1.000 | 0.400 | +0.600 |
| 96 | 0.133 | 0.000 | +0.133 |
| 88 | 0.000 | 0.000 | +0.000 |

### M1 wins (top 5):
| prompt_idx | B0 F1 | M1 F1 | delta |
|---|---|---|---|
| 93 | 0.000 | 1.000 | -1.000 |
| 117 | 0.214 | 0.261 | -0.047 |
| 88 | 0.000 | 0.000 | +0.000 |
| 94 | 1.000 | 1.000 | +0.000 |
| 97 | 0.000 | 0.000 | +0.000 |

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
| 143 | 1.000 | 0.000 | +1.000 |
| 151 | 1.000 | 0.000 | +1.000 |
| 155 | 1.000 | 0.000 | +1.000 |
| 152 | 0.308 | 0.000 | +0.308 |
| 138 | 0.000 | 0.000 | +0.000 |

### A4/cs1024 wins (top 5):
| prompt_idx | M4/cs1024 F1 | A4/cs1024 F1 | delta |
|---|---|---|---|
| 140 | 0.000 | 1.000 | -1.000 |
| 138 | 0.000 | 0.000 | +0.000 |
| 139 | 0.000 | 0.000 | +0.000 |
| 142 | 0.000 | 0.000 | +0.000 |
| 145 | 1.000 | 1.000 | +0.000 |

**Example: prompt 143 (M4/cs1024 wins)**

*M4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Forbidden Daughters`
- **Gold**: `Forbidden Daughters`

*A4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `The Third Kiss`
- **Gold**: `Forbidden Daughters`

**Example: prompt 140 (A4/cs1024 wins)**

*M4/cs1024* (F1=0.000, type=wrong):
- **Pred**: `Brooklyn`
- **Gold**: `Poznań`

*A4/cs1024* (F1=1.000, type=correct):
- **Pred**: `Poznań`
- **Gold**: `Poznań`
