import base64

with open('soup.modal.yaml', 'rb') as f:
    encoded = base64.b64encode(f.read()).decode('ascii')

# Split into 76-char chunks for proper Python string concatenation
chunks = [encoded[i:i+76] for i in range(0, len(encoded), 76)]

# Build the _CONFIG_B64 assignment
lines = ['_CONFIG_B64 = (']
for i, chunk in enumerate(chunks):
    lines.append(f'    "{chunk}"')
lines.append(')')

result = '\n'.join(lines)
print(result)
