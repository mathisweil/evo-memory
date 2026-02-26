from datasets import load_dataset

print("Loading Qasper from LongBench...")
data = load_dataset('THUDM/LongBench', 'qasper', split='test')

lengths = [d['length'] for d in data]

print(f"Total samples: {len(lengths)}")
print(f"Min length:    {min(lengths)} words")
print(f"Max length:    {max(lengths)} words")
print(f"Mean length:   {sum(lengths)/len(lengths):.0f} words")
print()
print("--- Coverage at different context lengths ---")
print(f"(approx tokens = words x 1.3)")
print()

thresholds = [
    (2000,  2600),
    (3000,  3900),
    (3150,  4096),
    (5000,  6500),
    (6000,  7800),
    (7500,  9750),
    (10000, 13000),
]

for words, tokens in thresholds:
    count = sum(1 for l in lengths if l < words)
    pct = count / len(lengths) * 100
    print(f"Under {words:>6} words (~{tokens:>5} tokens): {count:>3} / {len(lengths)}  ({pct:.1f}%)")