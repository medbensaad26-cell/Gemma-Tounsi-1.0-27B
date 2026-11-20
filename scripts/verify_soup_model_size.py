"""Verify Soup 0.73.3 model-size patch correctly detects Gemma 3 27B as 27B."""
from soup_cli.utils.gpu import model_size_from_name

model_id = "google/gemma-3-27b-it"

actual = model_size_from_name(model_id)

print(f"Model: {model_id}")
print(f"Detected size: {actual}B")

if actual != 27:
    raise SystemExit(
        f"Model-size verification failed: expected 27B, got {actual}B"
    )

print("SUCCESS: Gemma 3 27B model-size patch verified.")
