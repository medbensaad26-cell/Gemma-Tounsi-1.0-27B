#!/usr/bin/env python3
"""
scripts/modal_blackwell_smoke_test.py — Gemma Tounsi 1.0 (27B)

LIMITED GPU SMOKE TEST — runs ONCE, requests exactly one RTX-PRO-6000,
no GPU fallback, 600 s timeout, tiny locally-defined model (no Gemma 27B
download). Verifies Blackwell + bitsandbytes NF4 + PEFT LoRA compatibility
BEFORE any real training spend.

Purpose:
  - Report CUDA compute capability, driver, runtime, torch, bitsandbytes,
    transformers, PEFT, accelerate versions.
  - Test bitsandbytes NF4 4-bit linear forward + backward on a tiny model.
  - Test a small AdamW optimizer update.
  - Test PEFT LoRA integration.
  - Record peak GPU memory usage (non-fatal; never blocks compatibility tests).
  - Return explicit per-test PASS/FAIL and an overall PASS/FAIL.

Budget ceiling: intended total <= $1.00 (GPU $0.000842/s + CPU + RAM +
image build + keep-alive). Modal cannot strictly enforce a per-run dollar
cap; treat $1.00 as a planning target and stop the function immediately
after verification. Image build time is billed separately and is the
largest uncertain cost component — do not assume the $1.00 ceiling is
guaranteed enforceable.

EXECUTION RESTRICTION (preparation task):
  Do NOT run this script during this task. Launch only with explicit
  approval:  modal run --detach scripts/modal_blackwell_smoke_test.py

No automatic retries. No repeated GPU allocations. Stops after the first
successful verification.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

try:
    import modal
except ImportError:  # Allow importing pure functions for local linting
    modal = None


class _ModalStub:
    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    class Image:
        @staticmethod
        def debian_slim(python_version=None):
            return _ModalStub.Image()

        def pip_install(self, *args, **kwargs):
            return self

        def run_commands(self, *args, **kwargs):
            return self

    class Volume:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Volume()

    class Secret:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Secret()


if modal is None:
    modal = _ModalStub()

# --- Modal resources ---------------------------------------------------------
# Exactly one RTX-PRO-6000, NO fallback list, NO silent substitution.
GPU_IDENTIFIER = "RTX-PRO-6000"
SMOKE_TEST_TIMEOUT_SECONDS = 600  # strict execution ceiling

app = modal.App("gemma-tounsi-blackwell-smoke-test")

# Minimal image: only the packages needed to validate the Blackwell stack.
# No Soup, no Gemma 27B download.
smoke_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "bitsandbytes", "peft", "transformers", "accelerate")
    .run_commands(
        "python3 -c 'import torch; import bitsandbytes; import peft; "
        "import transformers; import accelerate; print(\"smoke deps OK\")'"
    )
)


def _stage(message: str) -> None:
    """Print a clear execution stage before each operation."""
    print(f"[stage] {message}")


def record_library_versions() -> Dict[str, Any]:
    """Import and report library versions without touching CUDA."""
    versions: Dict[str, Any] = {}
    import torch

    versions["torch_version"] = torch.__version__
    versions["cuda_version"] = getattr(torch.version, "cuda", None)
    for name in ("bitsandbytes", "peft", "transformers", "accelerate"):
        try:
            module = __import__(name)
            versions[f"{name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[f"{name}_version"] = f"ERROR: {type(exc).__name__}: {exc}"
    versions["python_version"] = sys.version
    return versions


def nvidia_smi_memory() -> str:
    """Independent GPU memory measurement via nvidia-smi, if available."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
        return f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()}"
    except FileNotFoundError:
        return "nvidia-smi not available"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi error: {type(exc).__name__}: {exc}"


def test_pytorch_cuda_ops(device: str) -> Dict[str, Any]:
    """Basic PyTorch CUDA operations: tensor create, matmul, add."""
    import torch

    results: Dict[str, Any] = {"test": "pytorch_cuda_ops", "pass": False}
    try:
        a = torch.randn(64, 64, device=device, dtype=torch.float32)
        b = torch.randn(64, 64, device=device, dtype=torch.float32)
        c = a @ b
        c = c + 1.0
        assert c.shape == (64, 64), f"unexpected output shape {c.shape}"
        assert c.device.type == "cuda", "result not on CUDA"
        results["pass"] = True
        results["detail"] = "CUDA tensor creation + matmul + add OK"
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"{type(exc).__name__}: {exc}"
    return results


def test_bitsandbytes_nf4_init(device: str) -> Dict[str, Any]:
    """bitsandbytes Linear4bit (NF4) initialization only (no forward/backward)."""
    import torch
    import bitsandbytes as bnb

    results: Dict[str, Any] = {"test": "bitsandbytes_nf4_init", "pass": False}
    try:
        layer = bnb.nn.Linear4bit(
            256, 128, compute_dtype=torch.bfloat16,
            compress_statistics=False, quant_type="nf4",
        ).to(device)
        assert layer.weight is not None, "NF4 weight is None"
        results["pass"] = True
        results["detail"] = "NF4 Linear4bit initialized on CUDA"
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"{type(exc).__name__}: {exc}"
    return results


def test_nf4_forward_backward(device: str) -> Dict[str, Any]:
    """NF4 Linear4bit forward pass + backward pass.

    bitsandbytes Params4bit weights are frozen (requires_grad=False) by
    design, so weight.grad is None. Gradients flow through the input
    tensor instead. This test verifies genuine backward propagation
    through the frozen NF4 layer and confirms the weight stays frozen.
    """
    import torch
    import bitsandbytes as bnb

    results: Dict[str, Any] = {"test": "bitsandbytes_nf4_forward_backward", "pass": False}
    try:
        layer = bnb.nn.Linear4bit(
            256, 128, compute_dtype=torch.bfloat16,
            compress_statistics=False, quant_type="nf4",
        ).to(device)

        # Quantized weight must be frozen.
        assert layer.weight.requires_grad is False, "NF4 weight should be frozen"
        assert layer.weight.grad is None, "NF4 weight grad should be None pre-backward"

        # Input requires grad so backward can propagate through the layer.
        x = torch.randn(4, 256, device=device, dtype=torch.bfloat16, requires_grad=True)

        # Forward.
        out = layer(x)
        assert out.shape == (4, 128), f"unexpected output shape {out.shape}"

        # Backward.
        loss = out.float().mean()
        loss.backward()

        # Gradients must flow to the input.
        assert x.grad is not None, "input grad is None"
        assert torch.isfinite(x.grad).all(), "input grad contains non-finite values"
        assert x.grad.abs().sum().item() > 0, "input grad is zero"

        # Quantized weight must remain frozen after backward.
        assert layer.weight.requires_grad is False, "NF4 weight became trainable"
        assert layer.weight.grad is None, "NF4 weight grad should remain None post-backward"

        results["pass"] = True
        results["detail"] = "NF4 forward + backward + input grad OK (weight frozen)"
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"{type(exc).__name__}: {exc}"
    return results


def test_adamw_update(device: str) -> Dict[str, Any]:
    """Small AdamW optimizer update on a CUDA parameter."""
    import torch

    results: Dict[str, Any] = {"test": "adamw_update", "pass": False}
    try:
        p = torch.nn.Parameter(torch.randn(64, device=device, dtype=torch.float32))
        opt = torch.optim.AdamW([p], lr=1e-3)
        loss = p.float().pow(2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        results["pass"] = True
        results["detail"] = "AdamW step executed without error"
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"{type(exc).__name__}: {exc}"
    return results


def test_peft_lora(device: str) -> Dict[str, Any]:
    """PEFT LoRA attachment + forward + backward + LoRA gradient update.

    Uses a toy model with explicitly named submodules (q_proj, v_proj) so
    target_modules match the modules that actually exist. Gemma training
    uses target_modules: "auto" (production config) — this toy model is
    separate and only validates the PEFT LoRA mechanism.

    PEFT initializes LoRA B to zero and LoRA A to random values. On the
    first backward pass, LoRA A's gradient is zero (d(output)/d(A) = B·x = 0
    because B = 0) while LoRA B's gradient is nonzero. This test accounts for
    that initialization behavior and verifies genuine LoRA learning across two
    optimizer steps.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    results: Dict[str, Any] = {"test": "peft_lora", "pass": False}
    try:
        # Deterministic seed.
        torch.manual_seed(42)

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 64)
                self.v_proj = nn.Linear(128, 64)

            def forward(self, x):
                return self.q_proj(x) + self.v_proj(x)

        base = ToyModel().to(device)
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                            target_modules=["q_proj", "v_proj"])
        model = get_peft_model(base, config)

        # At least one targeted module was adapted.
        adapted = any(
            "q_proj" in n or "v_proj" in n
            for n, _ in model.named_modules()
        )
        assert adapted, "no q_proj/v_proj module adapted"

        # Trainable LoRA parameters exist.
        lora_params = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        assert len(lora_params) > 0, "no trainable LoRA parameters"

        def _grad_info(p):
            if p.grad is None:
                return {"exists": False, "norm": 0.0, "finite": False, "nonzero": False}
            g = p.grad
            return {
                "exists": True,
                "norm": float(g.norm().item()),
                "finite": bool(torch.isfinite(g).all()),
                "nonzero": bool(g.abs().sum().item() > 0),
            }

        # Deterministic input and target.
        x = torch.randn(2, 128, device=device, dtype=torch.float32)
        target = torch.randn(2, 64, device=device, dtype=torch.float32)

        # ---- FIRST STEP ----
        # Save copies of all LoRA adapter parameters before the update.
        lora_snapshots = {
            n: p.detach().clone() for n, p in lora_params
        }

        # Forward pass through the LoRA-adapted model.
        out = model(x)
        assert out.shape == (2, 64), f"unexpected output shape {out.shape}"

        # MSE loss against a non-identical target; depends on LoRA output.
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        # Collect per-parameter gradient diagnostics for LoRA A and B.
        lora_a_entries = []
        lora_b_entries = []
        for n, p in lora_params:
            info = _grad_info(p)
            info["name"] = n
            if "lora_a" in n.lower():
                lora_a_entries.append(info)
            elif "lora_b" in n.lower():
                lora_b_entries.append(info)

        # At least one trainable LoRA parameter receives a finite, nonzero grad.
        any_valid_grad = any(
            e["exists"] and e["finite"] and e["nonzero"]
            for e in lora_a_entries + lora_b_entries
        )
        assert any_valid_grad, "no trainable LoRA parameter received a valid gradient"

        # Not all LoRA gradients are zero.
        all_zero = all(
            not (e["exists"] and e["nonzero"])
            for e in lora_a_entries + lora_b_entries
        )
        assert not all_zero, "all LoRA gradients are zero"

        # No NaN/inf in any existing gradient.
        for e in lora_a_entries + lora_b_entries:
            if e["exists"]:
                assert e["finite"], f"non-finite gradient in {e['name']}"

        # LoRA B should receive a gradient on the first step: B starts at zero,
        # A is random, so d(loss)/d(B) = d(loss)/d(out) · A·x != 0, while
        # d(loss)/d(A) = B^T · ... = 0 because B = 0.
        b_has_grad = any(e["exists"] and e["nonzero"] for e in lora_b_entries)
        assert b_has_grad, "LoRA B parameter did not receive a gradient on first step"

        # Optimizer update on trainable LoRA parameters.
        opt = torch.optim.AdamW([p for _, p in lora_params], lr=1e-3)
        opt.step()
        opt.zero_grad()

        # Verify at least one LoRA adapter parameter actually changed.
        param_changed = any(
            not torch.equal(lora_snapshots[n], p.detach())
            for n, p in lora_params
        )
        assert param_changed, "no LoRA adapter parameter changed after optimizer step"

        # ---- SECOND STEP ----
        x2 = torch.randn(2, 128, device=device, dtype=torch.float32)
        target2 = torch.randn(2, 64, device=device, dtype=torch.float32)

        out2 = model(x2)
        loss2 = F.mse_loss(out2.float(), target2.float())
        loss2.backward()

        # Re-collect gradient diagnostics on the second step.
        lora_a_entries2 = []
        lora_b_entries2 = []
        for n, p in lora_params:
            info = _grad_info(p)
            info["name"] = n
            if "lora_a" in n.lower():
                lora_a_entries2.append(info)
            elif "lora_b" in n.lower():
                lora_b_entries2.append(info)

        # On the second step, B is no longer zero, so both A and B gradients
        # should be finite and nonzero.
        for e in lora_a_entries2:
            assert e["exists"] and e["finite"] and e["nonzero"], \
                f"LoRA A gradient missing/non-finite/zero on second step: {e['name']}"
        for e in lora_b_entries2:
            assert e["exists"] and e["finite"] and e["nonzero"], \
                f"LoRA B gradient missing/non-finite/zero on second step: {e['name']}"

        results["pass"] = True
        results["detail"] = "PEFT LoRA attach + forward + backward + grad + two-step optimizer OK"
    except Exception as exc:  # noqa: BLE001
        results["error"] = f"{type(exc).__name__}: {exc}"
    return results


@app.function(
    image=smoke_image,
    gpu=GPU_IDENTIFIER,
    timeout=SMOKE_TEST_TIMEOUT_SECONDS,
    cpu=1,
    memory=4096,  # MiB — minimal reservation
)
def run_smoke_test() -> Dict[str, Any]:
    """Execute the Blackwell compatibility smoke test on one RTX-PRO-6000.

    Returns a dict with per-test PASS/FAIL and an overall result.
    Stops immediately after the first successful verification — no retries.
    Memory monitoring is advisory (PASS/WARNING/FAIL) and never blocks the
    mandatory compatibility tests.
    """
    import torch

    # Mandatory results; memory_monitoring is advisory.
    results: Dict[str, Any] = {
        "gpu_detection": "FAIL",
        "cuda_initialization": "FAIL",
        "pytorch_operations": "FAIL",
        "nf4_quantization": "FAIL",
        "forward_backward": "FAIL",
        "optimizer_update": "FAIL",
        "peft_lora": "FAIL",
        "memory_monitoring": "FAIL",
    }
    details: Dict[str, str] = {}
    env: Dict[str, Any] = {}

    cuda_device: str | None = None
    current_device_index: int | None = None

    # --- Stage 1: CUDA initialization + GPU detection -----------------------
    _stage(
        "cuda_init: torch.cuda.is_available(), device_count, init(), "
        "current_device, device properties"
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    env["cuda_available"] = cuda_available
    env["device_count"] = device_count
    env.update(record_library_versions())

    if cuda_available and device_count > 0:
        try:
            torch.cuda.init()
            current_device_index = int(torch.cuda.current_device())
            cuda_device = f"cuda:{current_device_index}"
            props = torch.cuda.get_device_properties(current_device_index)
            env["device_name"] = props.name
            env["major"] = int(props.major)
            env["minor"] = int(props.minor)
            env["compute_capability"] = f"{env['major']}.{env['minor']}"
            env["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            results["cuda_initialization"] = "PASS"
            details["cuda_initialization"] = (
                f"CUDA initialized; current device {current_device_index}: {props.name}"
            )
            # GPU detection includes Blackwell sm_120 verification.
            if env["compute_capability"] == "12.0":
                results["gpu_detection"] = "PASS"
                details["gpu_detection"] = (
                    f"detected {props.name} compute capability "
                    f"{env['compute_capability']} (sm_120)"
                )
            else:
                details["gpu_detection"] = (
                    f"detected {props.name} but compute capability is "
                    f"{env['compute_capability']}, expected 12.0 (sm_120)"
                )
        except Exception as exc:  # noqa: BLE001
            results["cuda_initialization"] = "FAIL"
            details["cuda_initialization"] = f"{type(exc).__name__}: {exc}"
            details["gpu_detection"] = "CUDA initialization failed"
    else:
        details["cuda_initialization"] = (
            f"cuda_available={cuda_available}, device_count={device_count}"
        )
        details["gpu_detection"] = "no CUDA device detected"

    # --- Stage 2: memory monitoring (non-fatal) ----------------------------
    _stage("memory: reset_peak_memory_stats on current CUDA device")
    memory_nvidia_smi = ""
    if cuda_device is not None:
        try:
            torch.cuda.reset_peak_memory_stats(current_device_index)
            results["memory_monitoring"] = "PASS"
            details["memory_monitoring"] = (
                f"peak memory stats reset on {cuda_device}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARNING] reset_peak_memory_stats({current_device_index}) failed: "
                f"{type(exc).__name__}: {exc}"
            )
            results["memory_monitoring"] = "WARNING"
            details["memory_monitoring"] = (
                f"WARNING: {type(exc).__name__}: {exc}; "
                "continuing with compatibility tests"
            )
        memory_nvidia_smi = nvidia_smi_memory()
        details["memory_nvidia_smi"] = memory_nvidia_smi
    else:
        details["memory_monitoring"] = "skipped: no CUDA device"

    # Memory baseline (best-effort, non-fatal).
    mem_before: float | None = None
    if cuda_device is not None and torch.cuda.is_available():
        try:
            mem_before = (
                torch.cuda.memory_allocated(current_device_index) / (1024 ** 3)
            )
        except Exception:  # noqa: BLE001
            mem_before = None

    # --- Stage 3: PyTorch CUDA operations ----------------------------------
    if cuda_device is not None:
        _stage("pytorch_operations: CUDA tensor create + matmul + add")
        r = test_pytorch_cuda_ops(cuda_device)
        results["pytorch_operations"] = "PASS" if r["pass"] else "FAIL"
        details["pytorch_operations"] = r.get("detail") or r.get("error", "")
    else:
        details["pytorch_operations"] = "skipped: no CUDA device"

    # --- Stage 4: bitsandbytes NF4 init ------------------------------------
    if cuda_device is not None:
        _stage("nf4_quantization: bitsandbytes Linear4bit(NF4) initialization")
        r = test_bitsandbytes_nf4_init(cuda_device)
        results["nf4_quantization"] = "PASS" if r["pass"] else "FAIL"
        details["nf4_quantization"] = r.get("detail") or r.get("error", "")
    else:
        details["nf4_quantization"] = "skipped: no CUDA device"

    # --- Stage 5: NF4 forward + backward -----------------------------------
    if cuda_device is not None:
        _stage("forward_backward: NF4 Linear4bit forward + backward")
        r = test_nf4_forward_backward(cuda_device)
        results["forward_backward"] = "PASS" if r["pass"] else "FAIL"
        details["forward_backward"] = r.get("detail") or r.get("error", "")
    else:
        details["forward_backward"] = "skipped: no CUDA device"

    # --- Stage 6: AdamW optimizer update -----------------------------------
    if cuda_device is not None:
        _stage("optimizer_update: AdamW step on CUDA parameter")
        r = test_adamw_update(cuda_device)
        results["optimizer_update"] = "PASS" if r["pass"] else "FAIL"
        details["optimizer_update"] = r.get("detail") or r.get("error", "")
    else:
        details["optimizer_update"] = "skipped: no CUDA device"

    # --- Stage 7: PEFT LoRA ------------------------------------------------
    if cuda_device is not None:
        _stage("peft_lora: PEFT LoRA attach + forward + backward + gradients")
        r = test_peft_lora(cuda_device)
        results["peft_lora"] = "PASS" if r["pass"] else "FAIL"
        details["peft_lora"] = r.get("detail") or r.get("error", "")
    else:
        details["peft_lora"] = "skipped: no CUDA device"

    # Memory after (best-effort, non-fatal).
    mem_after: float | None = None
    if cuda_device is not None and torch.cuda.is_available():
        try:
            mem_after = (
                torch.cuda.max_memory_allocated(current_device_index) / (1024 ** 3)
            )
        except Exception:  # noqa: BLE001
            mem_after = None

    mandatory = [
        "gpu_detection",
        "cuda_initialization",
        "pytorch_operations",
        "nf4_quantization",
        "forward_backward",
        "optimizer_update",
        "peft_lora",
    ]
    all_mandatory_pass = all(results[k] == "PASS" for k in mandatory)
    overall = "PASS" if all_mandatory_pass else "FAIL"

    # Build a summary containing ONLY JSON-compatible primitives. Tensors,
    # torch.device, CUDA property objects, dtypes, and exception objects are
    # never returned — they would force the local deserialization layer to
    # import torch (ModuleNotFoundError: No module named 'torch').
    sanitized_summary: Dict[str, Any] = {
        "overall": overall,
        "gpu_identifier": str(GPU_IDENTIFIER),
        "gpu_count": int(env.get("device_count", 0)),
        "compute_capability": str(env.get("compute_capability")) if env.get("compute_capability") is not None else None,
        "device_name": str(env.get("device_name")) if env.get("device_name") is not None else None,
        "total_memory_gb": float(env.get("total_memory_gb")) if env.get("total_memory_gb") is not None else None,
        "cuda_version": str(env.get("cuda_version")) if env.get("cuda_version") is not None else None,
        "torch_version": str(env.get("torch_version")) if env.get("torch_version") is not None else None,
        "bitsandbytes_version": str(env.get("bitsandbytes_version")) if env.get("bitsandbytes_version") is not None else None,
        "peft_version": str(env.get("peft_version")) if env.get("peft_version") is not None else None,
        "transformers_version": str(env.get("transformers_version")) if env.get("transformers_version") is not None else None,
        "accelerate_version": str(env.get("accelerate_version")) if env.get("accelerate_version") is not None else None,
        "python_version": str(env.get("python_version")) if env.get("python_version") is not None else None,
        "memory_before_gb": float(round(mem_before, 3)) if mem_before is not None else None,
        "memory_after_gb": float(round(mem_after, 3)) if mem_after is not None else None,
        "memory_nvidia_smi": str(memory_nvidia_smi) if memory_nvidia_smi is not None else None,
        "results": {k: str(v) for k, v in results.items()},
        "details": {k: str(v) for k, v in details.items()},
        "all_mandatory_pass": bool(all_mandatory_pass),
        "cuda_sm_120": bool(env.get("compute_capability") == "12.0"),
        "timeout_seconds": int(SMOKE_TEST_TIMEOUT_SECONDS),
    }

    # Validate serializability in the remote container BEFORE returning, so a
    # non-primitive object surfaces here (with torch available) rather than in
    # the local deserialization layer (without torch).
    json_summary = json.dumps(sanitized_summary)
    print(json_summary)
    return json_summary


@app.local_entrypoint()
def main() -> None:
    """Entry point: `modal run --detach scripts/modal_blackwell_smoke_test.py`.

    Runs the smoke test once on exactly one RTX-PRO-6000 and prints the
    PASS/FAIL result. Detached — safe to shut down the local PC while it runs.
    """
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — Blackwell smoke test")
    print("=" * 60)
    print(f"GPU: {GPU_IDENTIFIER} — exactly one, no fallback")
    print(f"Timeout: {SMOKE_TEST_TIMEOUT_SECONDS}s")
    print("Budget ceiling: intended total <= $1.00 (planning target)")
    print("=" * 60)

    result = run_smoke_test.remote()

    # The remote function returns a JSON string (primitives only), so the
    # local client can deserialize it without importing torch.
    if isinstance(result, str):
        result = json.loads(result)

    print(json.dumps(result, indent=2))

    print("-" * 60)
    print("Per-test results:")
    for key in (
        "gpu_detection",
        "cuda_initialization",
        "pytorch_operations",
        "nf4_quantization",
        "forward_backward",
        "optimizer_update",
        "peft_lora",
        "memory_monitoring",
    ):
        print(f"  {key}: {result['results'][key]}")
    print("-" * 60)

    if result["overall"] == "PASS":
        print("SMOKE TEST RESULT: PASS — Blackwell stack verified.")
    else:
        print("SMOKE TEST RESULT: FAIL — do NOT proceed to training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
