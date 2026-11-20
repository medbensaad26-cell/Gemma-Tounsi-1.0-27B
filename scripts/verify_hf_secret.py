import os
import modal

app = modal.App("gemma-hf-secret-check")

hf_secret = modal.Secret.from_name("gemma-hf")


@app.function(
    secrets=[hf_secret],
    cpu=1,
    gpu=None,
    timeout=60,
)
def verify_secret():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
    if token:
        os.environ["HF_TOKEN"] = token
        result = True
    else:
        result = False

    print(f"HF_TOKEN available: {result}")

    if not result:
        raise RuntimeError("HF_TOKEN is missing!")


@app.local_entrypoint()
def main():
    verify_secret.remote()