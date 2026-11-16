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
print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
print(f"Critical: google/gemma-3-27b-it -> {model_size_from_name('google/gemma-3-27b-it')}B")
assert model_size_from_name('google/gemma-3-27b-it') == 27, "CRITICAL: 27b-it must return 27, not 7"
