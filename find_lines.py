lines = open('src/data/retention.py', encoding='utf-8').read().split('\n')
for i in [895, 1016, 1105, 1203, 1208, 1214]:
    safe = lines[i].encode('ascii', errors='replace').decode('ascii')
    print(f'{i+1}: {safe[:100]}')
