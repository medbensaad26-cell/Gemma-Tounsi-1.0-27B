#!/usr/bin/env python3
"""Generate soup_modal_app.py with properly base64-embedded config."""

import base64
import sys
from pathlib import Path

# Read and base64-encode the Modal-specific config
with open("soup.modal.yaml", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

# Split into 76-char chunks for proper Python implicit string concatenation
chunks = [encoded[i:i + 76] for i in range(0, len(encoded), 76)]
config_b64_lines = ["_CONFIG_B64 = ("]
for chunk in chunks:
    config_b64_lines.append(f'    "{chunk}"')
config_b64_lines.append(")")
config_b64 = "\n".join(config_b64_lines)

# Read the template (everything except the _CONFIG_B64 assignment)
template_path = Path("scripts/soup_modal_app_template.py")
if not template_path.exists():
    print("ERROR: scripts/soup_modal_app_template.py not found", file=sys.stderr)
    sys.exit(1)

template = template_path.read_text(encoding="utf-8")

# Replace the placeholder
content = template.replace("__CONFIG_B64_PLACEHOLDER__", config_b64)

output_path = Path("soup_modal_app.py")
output_path.write_text(content, encoding="utf-8")
print(f"Generated {output_path} ({len(content)} bytes)")
