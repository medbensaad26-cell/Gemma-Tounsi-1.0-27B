import json

files = ['data/splits/train.jsonl', 'data/splits/validation.jsonl', 'data/splits/test.jsonl']

for fname in files:
    print(f'=== {fname} ===')
    with open(fname, 'r', encoding='utf-8') as f:
        line = f.readline()
    obj = json.loads(line)
    for k, v in obj.items():
        if k == 'messages':
            print(f'  {k}: {len(v)} messages')
            for i, m in enumerate(v):
                role = m.get('role', 'N/A')
                content_len = len(m.get('content', ''))
                print(f'    msg {i}: role={role}, content_len={content_len}')
        else:
            print(f'  {k}: {v}')
    print()

# Check all records for schema consistency
print('=== Schema consistency check ===')
keys_seen = set()
for fname in files:
    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys_seen.update(obj.keys())
            # Check messages structure
            for m in obj.get('messages', []):
                assert 'role' in m, f'Missing role in {fname} line {i}'
                assert 'content' in m, f'Missing content in {fname} line {i}'
    print(f'  {fname}: OK ({i+1} lines checked)')
print(f'All keys seen: {sorted(keys_seen)}')
