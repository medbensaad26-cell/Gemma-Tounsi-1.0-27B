#!/usr/bin/env python3
"""Disposable test: Soup 0.77.2 compatibility with Gemma 3 27B.
Run in a Python 3.11 container (isolated, does not touch project env)."""
import json
import os
import subprocess
import sys

print("=" * 60)
print("Phase 3: Soup 0.77.2 isolated compatibility test")
print("=" * 60)

# 1. Check Python version
print(f"\nPython version: {sys.version}")
assert sys.version_info >= (3, 11), "Need Python 3.11+"
assert sys.version_info < (3, 14), "Need Python <3.14"
print("Python version OK: 3.11-3.13")

# 2. Install soup-cli 0.77.2
print("\n--- Installing soup-cli 0.77.2 ---")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "soup-cli==0.77.2"],
    capture_output=True, text=True, timeout=300
)
print(result.stdout[-2000:] if result.stdout else "(no stdout)")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(1)
print("Soup 0.77.2 installed successfully")

# 3. Verify version
print("\n--- Soup version ---")
result = subprocess.run(["soup", "version"], capture_output=True, text=True, timeout=30)
print(result.stdout.strip())
if result.stderr:
    print("STDERR:", result.stderr.strip())

# 4. Test model_size_from_name
print("\n--- Model size detection test ---")
test_script = '''
from soup_cli.utils.gpu import model_size_from_name
tests = [
    ("google/gemma-3-27b-it", 27),
    ("google/gemma-3-7b-it", 7),
    ("meta-llama/Llama-3.1-8B", 8),
    ("meta-llama/Llama-3.1-70B", 70),
    ("Qwen/Qwen2.5-14B", 14),
    ("Qwen/Qwen2.5-72B", 72),
    ("google/gemma-3-4b-it", 4),
    ("google/gemma-3-1b-it", 1),
    ("bigscience/bloom-17b", 17),
    ("facebook/opt-1.7b", 1.7),
]
all_ok = True
for name, expected in tests:
    got = model_size_from_name(name)
    ok = got == expected
    all_ok = all_ok and ok
    status = "OK" if ok else "FAIL"
    print(f"  {status}: {name} -> {got}B (expected {expected}B)")
print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'} (critical: 27b-it -> {model_size_from_name('google/gemma-3-27b-it')}B)')
assert model_size_from_name('google/gemma-3-27b-it') == 27, "CRITICAL: 27b-it must return 27, not 7"
'''
result = subprocess.run(
    [sys.executable, "-c", test_script],
    capture_output=True, text=True, timeout=60
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# 5. Load existing soup.yaml
print("\n--- Loading existing soup.yaml ---")
soup_yaml = os.path.join(os.getcwd(), "soup.yaml")
if os.path.exists(soup_yaml):
    result = subprocess.run(
        ["soup", "doctor", "--config", soup_yaml],
        capture_output=True, text=True, timeout=120
    )
    print("STDOUT:", result.stdout[-3000:] if result.stdout else "(none)")
    if result.stderr:
        print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    print(f"Return code: {result.returncode}")
else:
    print("soup.yaml not found - skipping doctor test")

# 6. Check Modal/cloud command
print("\n--- Cloud command check ---")
result = subprocess.run(["soup", "train", "--cloud", "--help"], capture_output=True, text=True, timeout=30)
stdout = result.stdout
if "modal" in stdout.lower() or "runpod" in stdout.lower() or "lambda" in stdout.lower():
    print("Cloud providers found in help:")
    for line in stdout.split("\n"):
        if any(p in line.lower() for p in ["modal", "runpod", "lambda", "cloud"]):
            print(f"  {line.strip()}")
else:
    print("No cloud providers found in --help output")
    print(stdout[:2000])

print("\n--- Test complete ---")
