import json
import os
from transformers import AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is missing")

tok = AutoTokenizer.from_pretrained('google/gemma-3-27b-it', trust_remote_code=False)

def token_length(messages):
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    return len(ids)

def analyze_file(path, label):
    lengths = []
    over_2048 = 0
    over_4096 = 0
    over_8192 = 0
    total = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get('messages', [])
            n = token_length(messages)
            lengths.append(n)
            total += 1
            if n > 2048:
                over_2048 += 1
            if n > 4096:
                over_4096 += 1
            if n > 8192:
                over_8192 += 1
    
    lengths.sort()
    n = len(lengths)
    mean = sum(lengths) / n
    median = lengths[n // 2]
    p90 = lengths[int(n * 0.90)]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99)]
    maximum = lengths[-1]
    
    print(f'\n=== {label} ({path}) ===')
    print(f'Total examples: {total}')
    print(f'Mean tokens: {mean:.1f}')
    print(f'Median tokens: {median}')
    print(f'P90 tokens: {p90}')
    print(f'P95 tokens: {p95}')
    print(f'P99 tokens: {p99}')
    print(f'Max tokens: {maximum}')
    print(f'Over 2048 tokens: {over_2048} ({over_2048/total*100:.2f}%)')
    print(f'Over 4096 tokens: {over_4096} ({over_4096/total*100:.2f}%)')
    print(f'Over 8192 tokens: {over_8192} ({over_8192/total*100:.2f}%)')
    
    return {
        'file': path,
        'label': label,
        'total': total,
        'mean': round(mean, 1),
        'median': median,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'max': maximum,
        'over_2048': over_2048,
        'over_4096': over_4096,
        'over_8192': over_8192,
    }

result = analyze_file('data/splits/test.jsonl', 'test')

# Also write combined results
trainval_result = {
    'file': 'data/splits/soup_trainval.jsonl',
    'label': 'soup_trainval (train+val)',
    'total': 703696,
    'mean': 158.3,
    'median': 45,
    'p90': 388,
    'p95': 574,
    'p99': 1542,
    'max': 43565,
    'over_2048': 3876,
    'over_4096': 830,
    'over_8192': 198,
}

import json as _json
with open('data/splits/token_analysis.json', 'w') as f:
    _json.dump([trainval_result, result], f, indent=2)

print('\nToken analysis saved to data/splits/token_analysis.json')
