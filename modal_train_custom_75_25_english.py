"""Modal application for Gemma Tounsi 1.0 (27B) QLoRA training — custom_75_25_english experiment.

All 8 deployment blockers resolved:
  1. Model-size patch applied in remote Modal image (soup-cli 0.73.3)
  2. HF Secret "gemma-hf" attached for HF_TOKEN
  3. Persistent dataset Volume "gemma-tounsi-data" mounted at /mnt/data
  4. Persistent checkpoint Volume "gemma-tounsi-checkpoints" mounted at /checkpoints
     with checkpoint-aware commits (only after trainer_state.json is written)
  5. Persistent HF cache Volume "gemma-tounsi-hf-cache" mounted at /hf-cache
     committed after model download completes
  6. Explicit resume from the newest complete checkpoint (minimum Step 900)
  7. Runtime timeout and budget protection (MAX_RUNTIME_SECONDS, STOP_AFTER_STEPS)
  8. All paths are absolute persistent paths

Run with: modal run --detach modal_train_custom_75_25_english.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import modal
except ImportError:  # Allow importing pure functions for local testing
    modal = None


class _ModalStub:
    """Stub for modal.* when the modal package is not installed locally."""

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

        def apt_install(self, *args, **kwargs):
            return self

        def add_local_file(self, *args, **kwargs):
            return self

        def run_commands(self, *args, **kwargs):
            return self

    class Volume:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Volume()

        def with_mount_options(self, **kwargs):
            return self

    class Secret:
        @staticmethod
        def from_name(*args, **kwargs):
            return _ModalStub.Secret()


if modal is None:
    modal = _ModalStub()

# --- Embedded training config (soup.modal.yaml) ------------------------------
# Base64-encoded Modal-specific configuration. Uses absolute persistent paths:
#   data.train  -> /mnt/data/soup_trainval.jsonl  (Volume: gemma-tounsi-data)
#   output      -> /checkpoints/gemma-tounsi-1.0-27b  (Volume: gemma-tounsi-checkpoints)
_CONFIG_B64 = (
    "IyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PQojIHNvdXAuY3VzdG9tXzc1XzI1X2VuZ2xpc2gubW9kYWwu"
    "eWFtbCAtIEdlbW1hIFRvdW5zaSAxLjAgKDI3QikgTW9kYWwgcmVtb3RlIGNvbmZpZwojID09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09CiMKIyBNb2RhbC1zcGVjaWZpYyBjb25maWd1cmF0aW9uIGZvciB0aGUg"
    "Y3VzdG9tXzc1XzI1X2VuZ2xpc2ggZXhwZXJpbWVudC4KIyBQYXRocyBhcmUgYWJzb2x1dGUgY29u"
    "dGFpbmVyIHBhdGhzIGluc2lkZSB0aGUgTW9kYWwgY29udGFpbmVyOgojICAgLSBEYXRhc2V0OiAv"
    "bW50L2RhdGEvY3VzdG9tXzc1XzI1X2VuZ2xpc2gvdHJhaW4uanNvbmwgIChWb2x1bWU6IGdlbW1h"
    "LXRvdW5zaS1kYXRhKQojICAgLSBPdXRwdXQ6ICAvY2hlY2twb2ludHMvZ2VtbWEtdG91bnNpLTEu"
    "MC0yN2ItY3VzdG9tLTc1LTI1LWVuZ2xpc2ggIChWb2x1bWU6IGdlbW1hLXRvdW5zaS1jaGVja3Bv"
    "aW50cykKIyAgIC0gSEYgY2FjaGU6IC9oZi1jYWNoZSAgKFZvbHVtZTogZ2VtbWEtdG91bnNpLWhm"
    "LWNhY2hlKQojCiMgVGhpcyBmaWxlIGlzIGVtYmVkZGVkIChiYXNlNjQpIGludG8gbW9kYWxfdHJh"
    "aW5fY3VzdG9tXzc1XzI1X2VuZ2xpc2gucHkuCiMgRG8gTk9UIGVkaXQgdGhpcyBmaWxlIGRpcmVj"
    "dGx5OyByZWdlbmVyYXRlIGFuZCByZS1lbWJlZCBpZiBoeXBlcnBhcmFtZXRlcnMgY2hhbmdlLgoj"
    "ID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09CgojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CiMgc291cC5jdXN0"
    "b21fNzVfMjVfZW5nbGlzaC55YW1sIOKAlCBHZW1tYSBUb3Vuc2kgMS4wICgyN0IpIFFMb1JBIGV4"
    "cGVyaW1lbnRhbCBjb25maWcKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQojCiMgRVhQRVJJTUVOVEFM"
    "IENPTkZJR1VSQVRJT04g4oCUIGRvZXMgTk9UIG1vZGlmeSBzb3VwLnlhbWwgKHByb2R1Y3Rpb24p"
    "LgojCiMgRGF0YXNldDogZGF0YS9leHBlcmltZW50cy9kb3duc2FtcGxlZC9jdXN0b21fNzVfMjVf"
    "ZW5nbGlzaC90cmFpbi5qc29ubAojICAgLSAzODQsNTcyIGV4YW1wbGVzLCA0NSw5MjYsMjgyIHRv"
    "a2VucyAodmVyaWZpZWQsIHNlZSBtYW5pZmVzdC5qc29uKQojICAgLSBBcmFiaXppIDE4Myw0MjMg"
    "LyBhciAxNjEsODM3IC8gbWl4ZWQgMTEsOTA1IC8gZW4gMjcsNDA3CiMgICAtIFB1cmUgdHJhaW5p"
    "bmcgc3BsaXQ6IGNvbnRhaW5zIE5PIHZhbGlkYXRpb24uanNvbmwgYW5kIE5PIHRlc3QuanNvbmwg"
    "ZXhhbXBsZXMKIyAgICAgKGludGVncml0eSB2ZXJpZmllZDogbm8gdmFsaWRhdGlvbi90ZXN0IGxl"
    "YWthZ2UpLgojCiMgVmFsaWRhdGlvbjogZGF0YS9zcGxpdHMvdmFsaWRhdGlvbi5qc29ubCAoMzcs"
    "MDM2IGV4YW1wbGVzKSBpcyB0aGUgb3JpZ2luYWwKIyAgIDUlIHZhbGlkYXRpb24gc3BsaXQgYW5k"
    "IGlzIHByZXNlcnZlZCBVTkNIQU5HRUQuCiMgVGVzdDogZGF0YS9zcGxpdHMvdGVzdC5qc29ubCAo"
    "MzcsMDM4IGV4YW1wbGVzKSByZW1haW5zIGV4Y2x1ZGVkLgojCiMgSU1QT1JUQU5UIOKAlCBTb3Vw"
    "IDAuNzMuMyB2YWxpZGF0aW9uIGxpbWl0YXRpb246CiMgICBTb3VwIDAuNzMuMyBEYXRhQ29uZmln"
    "IGRvZXMgTk9UIHN1cHBvcnQgYSBzZXBhcmF0ZSBsb2NhbCB2YWxpZGF0aW9uIGZpbGUuCiMgICB2"
    "YWxfc3BsaXQgaXMgYXBwbGllZCBJTlRFUk5BTExZIHRvIHRoZSB0cmFpbiBmaWxlIG9ubHkuIEJl"
    "Y2F1c2UgdGhpcwojICAgY29uZmlnIHRyYWlucyBvbiBhIHB1cmUtdHJhaW5pbmcgZmlsZSAobm8g"
    "dmFsaWRhdGlvbiBleGFtcGxlcyBtaXhlZCBpbiksCiMgICB2YWxfc3BsaXQgaXMgc2V0IHRvIDAu"
    "MCBzbyAxMDAlIG9mIHRoZSBjdXN0b20gdHJhaW4gZGF0YSBpcyB1c2VkIGZvcgojICAgdHJhaW5p"
    "bmcgYW5kIE5PIGV4YW1wbGVzIGFyZSB3aXRoaGVsZCBmb3IgYW4gaW50ZXJuYWwgdmFsaWRhdGlv"
    "biBzcGxpdC4KIyAgIGRhdGEvc3BsaXRzL3ZhbGlkYXRpb24uanNvbmwgaXMgdGhlcmVmb3JlIE5P"
    "VCBjb25zdW1lZCBieSBTb3VwIGR1cmluZwojICAgdHJhaW5pbmc7IGl0IGlzIHJlc2VydmVkIGZv"
    "ciBwb3N0LXRyYWluaW5nIGV2YWx1YXRpb24uIFRoaXMgZ3VhcmFudGVlcwojICAgemVybyB2YWxp"
    "ZGF0aW9uL3Rlc3QgbGVha2FnZSBpbnRvIHRyYWluaW5nIGJ5IGNvbnN0cnVjdGlvbi4KIwojIFFM"
    "b1JBIGh5cGVycGFyYW1ldGVycyBtaXJyb3Igc291cC55YW1sIChubyBhcmJpdHJhcnkgY2hhbmdl"
    "cyk6CiMgICA0LWJpdCBxdWFudGl6YXRpb24sIExvUkEgcj0xNi9hbHBoYT0zMi9kcm9wb3V0PTAu"
    "MDUsCiMgICB0YXJnZXRfbW9kdWxlcz1bcV9wcm9qLCBrX3Byb2osIHZfcHJvaiwgb19wcm9qXSAo"
    "ZXhwbGljaXQ7ICJhdXRvIiBpcwojICAgY29udmVydGVkIHRvIE5vbmUgYnkgU291cCAwLjczLjMg"
    "YW5kIHJlamVjdGVkIGJ5IFBFRlQpLAojICAgYmF0Y2hfc2l6ZT0xLCBncmFkaWVudF9hY2N1bXVs"
    "YXRpb25fc3RlcHM9OCwgbHI9MS4wZS01LCBlcG9jaHM9MiwKIyAgIG1heF9sZW5ndGg9ODE5Miwg"
    "Z3JhZGllbnRfY2hlY2twb2ludGluZz10cnVlLCBzYXZlX3N0ZXBzPTEwMC4KIwojIFJ1biBsb2Nh"
    "bGx5IChEb2NrZXIpOgojICAgZG9ja2VyIGNvbXBvc2UgcnVuIC0tcm0gc291cCBzb3VwIHRyYWlu"
    "IC0tY29uZmlnIC93b3Jrc3BhY2Uvc291cC5jdXN0b21fNzVfMjVfZW5nbGlzaC55YW1sCiMKIyBQ"
    "YXRocyBhcmUgY29udGFpbmVyIHBhdGhzOiB0aGUgcmVwb3NpdG9yeSByb290IGlzIG1vdW50ZWQg"
    "YXQgL3dvcmtzcGFjZS4KIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQoKIyAtLS0gQmFzZSBtb2RlbCAt"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLQpiYXNlOiBnb29nbGUvZ2VtbWEtMy0yN2ItaXQKCiMgU291cENvbmZpZy50YXNrIOKAlCBz"
    "dXBlcnZpc2VkIGZpbmUtdHVuaW5nLgp0YXNrOiBzZnQKCiMgU291cENvbmZpZy5iYWNrZW5kIOKA"
    "lCBzdG9jayB0cmFuc2Zvcm1lcnMgcGF0aC4KYmFja2VuZDogdHJhbnNmb3JtZXJzCgojIFNvdXBD"
    "b25maWcubW9kYWxpdHkg4oCUIHRleHQgU0ZULgptb2RhbGl0eTogdGV4dAoKIyAtLS0gRGF0YSAt"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLQpkYXRhOgogIHRyYWluOiAvbW50L2RhdGEvY3VzdG9tXzc1XzI1X2VuZ2xpc2gv"
    "dHJhaW4uanNvbmwKICBmb3JtYXQ6IGNoYXRtbAogICMgU291cCAwLjczLjMgY2FydmVzIHZhbGlk"
    "YXRpb24gaW50ZXJuYWxseSBmcm9tIHRoZSB0cmFpbiBmaWxlLiBTZXQgdG8gMC4wCiAgIyBzbyB0"
    "aGUgZW50aXJlIHB1cmUtdHJhaW5pbmcgY3VzdG9tIGZpbGUgaXMgdXNlZCBmb3IgdHJhaW5pbmcg"
    "YW5kIG5vCiAgIyBleGFtcGxlcyBhcmUgd2l0aGhlbGQuIHZhbGlkYXRpb24uanNvbmwgaXMgdXNl"
    "ZCBwb3N0LXRyYWluaW5nIG9ubHkuCiAgdmFsX3NwbGl0OiAwLjAKICBtYXhfbGVuZ3RoOiA4MTky"
    "CiAgdHJhaW5fb25fcmVzcG9uc2VzX29ubHk6IHRydWUKICBjaGF0X3RlbXBsYXRlOiBnZW1tYTMK"
    "CiMgLS0tIFRyYWluaW5nIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KdHJhaW5pbmc6CiAgZXBvY2hzOiAyCiAgbHI6IDEuMGUt"
    "NQogIGJhdGNoX3NpemU6IDEKICBncmFkaWVudF9hY2N1bXVsYXRpb25fc3RlcHM6IDgKCiAgIyBR"
    "TG9SQSA9IDQtYml0IHF1YW50aXplZCBiYXNlICsgTG9SQSBhZGFwdGVyLgogIHF1YW50aXphdGlv"
    "bjogNGJpdAoKICAjIExvUkEgYWRhcHRlciAoZXhwbGljaXQgR2VtbWEgMyBhdHRlbnRpb24gdGFy"
    "Z2V0cyDigJQgU291cCAwLjczLjMgY29udmVydHMKICAjICJhdXRvIiB0byBOb25lIGJlZm9yZSBQ"
    "RUZULCB3aGljaCByYWlzZXMgIlBsZWFzZSBzcGVjaWZ5IHRhcmdldF9tb2R1bGVzIG9yCiAgIyB0"
    "YXJnZXRfcGFyYW1ldGVycyBpbiBwZWZ0X2NvbmZpZyI7IHNlZSBzZnQucHkgdGFyZ2V0X21vZHVs"
    "ZSBoYW5kbGluZykuCiAgbG9yYToKICAgIHI6IDE2CiAgICBhbHBoYTogMzIKICAgIGRyb3BvdXQ6"
    "IDAuMDUKICAgIHRhcmdldF9tb2R1bGVzOgogICAgICAtIHFfcHJvagogICAgICAtIGtfcHJvagog"
    "ICAgICAtIHZfcHJvagogICAgICAtIG9fcHJvagoKICBncmFkaWVudF9jaGVja3BvaW50aW5nOiB0"
    "cnVlCiAgdHJhaW5fb25fZW90OiB0cnVlCgogIHNlZWQ6IDQyCiAgbG9nZ2luZ19zdGVwczogMTAK"
    "ICBzYXZlX3N0ZXBzOiAxMDAKCiAgc3RyZWFtX2xheWVyczogZmFsc2UKCiMgLS0tIE91dHB1dCAt"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0Kb3V0cHV0OiAvY2hlY2twb2ludHMvZ2VtbWEtdG91bnNpLTEuMC0yN2ItY3VzdG9t"
    "LTc1LTI1LWVuZ2xpc2g="
)

# --- Runtime limits ----------------------------------------------------------
# 23 hours with 1 hour margin before Modal's 24-hour hard timeout.
# Training stops at a completed checkpoint, commits it, and exits cleanly.
MAX_RUNTIME_SECONDS: int = 82800  # 23 hours (training soft-stop)
MODAL_FUNCTION_TIMEOUT_SECONDS: int = 86400  # 24 hours (Modal hard timeout)
STOP_AFTER_STEPS: Optional[int] = None  # Set to stop after N completed steps
MIN_RESUME_STEP: int = 900  # Do not cold-start below the verified recovery point
# Reserve time before Modal's hard timeout for final Volume commits/manifest.
FINALIZATION_RESERVE_SECONDS: int = 1800  # 30 minutes
PROCESS_POLL_INTERVAL_SECONDS: int = 10

# --- Budget-aware early stopping -------------------------------------------
# Modal pricing (verified Nov 18 2026):
#   GPU:    $0.000842 / second = $3.0312 / GPU-hour
#   CPU:    $0.0000131 / second per core = $0.04716 / CPU-hour
#   RAM:    $0.00000222 / second per GiB = $0.007992 / GiB-hour
# The training container requests 1 GPU, 1 CPU core, 4 GiB RAM.
# CPU + RAM add ~2.6% overhead on top of the GPU-only rate.
# A GPU-only timer is NOT a guaranteed total-cost cap; the all-in rate below
# includes CPU and RAM. Image build is billed separately (one-time, buffered).
GPU_COST_PER_SECOND: float = 0.000842
CPU_COST_PER_SECOND: float = 0.0000131  # 1 core
RAM_COST_PER_SECOND: float = 0.00000888  # 4 GiB × $0.00000222
ALL_IN_COST_PER_SECOND: float = (
    GPU_COST_PER_SECOND + CPU_COST_PER_SECOND + RAM_COST_PER_SECOND
)
TRAINING_BUDGET_USD: float = 24.00  # hard ceiling for training spend
BUDGET_SAFETY_MARGIN_USD: float = 0.50  # stop before hitting the ceiling
IMAGE_BUILD_BUFFER_USD: float = 2.00  # one-time image build, billed separately
# Estimated seconds until the training budget is nearly exhausted (all-in):
#   (24.00 - 0.50 - 2.00) / 0.00086398 = 24,874 s (~6.91 hr)
BUDGET_STOP_SECONDS: float = (
    (TRAINING_BUDGET_USD - BUDGET_SAFETY_MARGIN_USD - IMAGE_BUILD_BUFFER_USD)
    / ALL_IN_COST_PER_SECOND
)

# --- Volume names and mount paths --------------------------------------------
DATA_VOLUME_NAME = "gemma-tounsi-data"
CHECKPOINT_VOLUME_NAME = "gemma-tounsi-checkpoints"
HF_CACHE_VOLUME_NAME = "gemma-tounsi-hf-cache"

DATA_MOUNT = "/mnt/data"
CHECKPOINT_MOUNT = "/checkpoints"
HF_CACHE_MOUNT = "/hf-cache"

DATASET_PATH = f"{DATA_MOUNT}/custom_75_25_english/train.jsonl"
VAL_DATASET_PATH = f"{DATA_MOUNT}/validation.jsonl"
DATASET_SHA256_FILE = f"{DATA_MOUNT}/custom_75_25_english/train.jsonl.sha256"
OUTPUT_DIR = f"{CHECKPOINT_MOUNT}/gemma-tounsi-1.0-27b-custom-75-25-english"
CHECKPOINT_MANIFEST_PATH = f"{OUTPUT_DIR}/checkpoint_manifest.json"
LATEST_VALID_CHECKPOINT_PATH = f"{OUTPUT_DIR}/latest-valid-checkpoint"

# --- Preflight dataset fingerprints (verified on Modal Volume mount) ---
# These values match the successful remote CPU Volume mount preflight.
# The remote CPU-only preflight checks the mounted Volume copies against
# these before allocating the RTX PRO 6000. Datasets are never downloaded
# to the local machine during training launch.
EXPECTED_TRAIN_SIZE: int = 293746534
EXPECTED_TRAIN_SHA256: str = (
    "b6369becf9d536495df59a0849065f5117903596038eff7a9945168ca4672ff0"
)
EXPECTED_VAL_SIZE: int = 35166425
EXPECTED_VAL_SHA256: str = (
    "b57b3bb87999c6f58c656d1b25db747267ef26913b32be0f6610f716451060d4"
)

# Remote CPU-only preflight resources (no GPU, lightweight image).
PREFLIGHT_CPU: float = 0.25
PREFLIGHT_MEMORY_MIB: int = 512
PREFLIGHT_TIMEOUT_SECONDS: int = 300

# HF Trainer writes checkpoint files in this order; trainer_state.json is last.
# A checkpoint is considered complete ONLY when trainer_state.json exists.
CHECKPOINT_REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
)

# --- Modal resources ---------------------------------------------------------
app = modal.App("gemma-tounsi-custom-train")

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME, create_if_missing=True
)

# Commit lock prevents concurrent commits to the same Volume.
commit_lock = threading.Lock()

# Steps already committed — avoids re-committing unchanged checkpoints.
committed_steps: set[int] = set()


# BLOCKER 1: Apply model-size compatibility patch in the remote Modal image.
# The patch copies the repository-managed patch file and apply script into the
# image, runs them after soup-cli installation, and smoke-tests that
# model_size_from_name("google/gemma-3-27b-it") returns 27 (not 7).
#
# Python 3.12 is required because Soup 0.73.3 requires Python >=3.10,<3.13.
# The default Modal debian_slim image ships an incompatible Python version,
# which caused the image build to fail with "No matching distribution found
# for soup-cli==0.73.3".
def get_training_image() -> "modal.Image":
    """Build and return the Modal training image.

    Reused by both soup_modal_app.py (training entrypoint) and
    scripts/build_modal_image.py (build-only verification).
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        # System dependencies required by Soup / PyTorch training stack.
        .apt_install("curl", "ca-certificates")
        # Verify Python 3.12 is active before installing Python packages.
        .run_commands("python3 --version")
        .run_commands(
            "python3 -c 'import sys; assert sys.version_info >= (3, 10), "
            "\"Python too old\"; assert sys.version_info < (3, 13), "
            "\"Python too new\"; print(\"Python OK: \", sys.version)'"
        )
        .pip_install("soup-cli[train]==0.73.3")
        # Verify Soup installation and version.
        .run_commands("python3 -m pip show soup-cli")
        # Verify dependency consistency (soup-cli[train] extra already pulls
        # torch, transformers, peft, bitsandbytes, accelerate).
        .run_commands("python3 -m pip check")
        .add_local_file(
            "patches/soup-0.73.3-model-size.patch",
            remote_path="/tmp/soup-0.73.3-model-size.patch",
            copy=True,
        )
        .add_local_file(
            "scripts/apply_soup_compat_patch.py",
            remote_path="/tmp/apply_soup_compat_patch.py",
            copy=True,
        )
        .add_local_file(
            "scripts/verify_soup_model_size.py",
            remote_path="/tmp/verify_soup_model_size.py",
            copy=True,
        )
        .run_commands("python3 /tmp/apply_soup_compat_patch.py")
        # Verify patched model-size detection returns 27B.
        .run_commands("python3 /tmp/verify_soup_model_size.py")
        # Verify all required training packages are importable.
        .run_commands(
            "python3 -c 'import torch; import transformers; import peft; "
            "import bitsandbytes; import accelerate; import soup_cli; "
            "print(\"All required packages importable\")'"
        )
    )


image = get_training_image()

# Lightweight image for remote CPU-only dataset preflight.
# Intentionally excludes the Soup / PyTorch training stack.
preflight_image = modal.Image.debian_slim(python_version="3.12")

# BLOCKER 2: HF Secret "gemma-hf" provides HF_TOKEN. Attached to the function.
# The secret must be created manually: modal secret create gemma-hf HF_TOKEN=...
hf_secret = modal.Secret.from_name("gemma-hf")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model_revision() -> str:
    """Get the exact Hugging Face model revision/commit hash.

    Returns the resolved revision (commit SHA) for the gated base model.
    Falls back to "unknown" if the Hugging Face Hub is unreachable.
    """
    try:
        from huggingface_hub import model_info

        info = model_info("google/gemma-3-27b-it")
        return info.sha or info.revision or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def validate_dataset() -> str:
    """Validate dataset file exists and optionally verify SHA256.

    Returns the actual SHA256 of the dataset file.
    Raises FileNotFoundError if the dataset is missing.
    """
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Upload custom_75_25_english/train.jsonl to the "
            f"{DATA_VOLUME_NAME} Volume first (scripts/upload_modal_data.ps1)."
        )

    actual_sha256 = compute_sha256(dataset_path)

    # Optional SHA256 validation against precomputed fingerprint
    sha256_file = Path(DATASET_SHA256_FILE)
    if sha256_file.exists():
        expected = sha256_file.read_text(encoding="utf-8").strip()
        if actual_sha256 != expected:
            raise ValueError(
                f"Dataset SHA256 mismatch: expected {expected}, got {actual_sha256}"
            )
        print(f"Dataset SHA256 validated: {actual_sha256}")
    else:
        # No precomputed fingerprint yet; compute and store for future validation
        sha256_file.write_text(actual_sha256 + "\n", encoding="utf-8")
        print(f"Dataset SHA256 computed and stored: {actual_sha256}")

    return actual_sha256


def validate_hf_token() -> None:
    """Validate HF_TOKEN is available from the attached Secret.

    Checks HF_TOKEN first, then falls back to HF_Token (case-sensitive).
    If either contains a non-empty token, normalizes it into
    os.environ["HF_TOKEN"] for subprocess consumption.

    Raises RuntimeError if neither key contains a token.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
    if token:
        os.environ["HF_TOKEN"] = token
        return
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Ensure the 'gemma-hf' Modal Secret is attached to this function "
        "(modal secret create gemma-hf HF_TOKEN=...)."
    )


def _sha256_file_incremental(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 with incremental reads (no full file in RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=preflight_image,
    cpu=PREFLIGHT_CPU,
    memory=PREFLIGHT_MEMORY_MIB,
    timeout=PREFLIGHT_TIMEOUT_SECONDS,
    volumes={
        DATA_MOUNT: data_volume.with_mount_options(read_only=True),
    },
)
def run_dataset_preflight() -> Dict[str, Any]:
    """Remote CPU-only dataset preflight inside a Modal container.

    Mounts gemma-tounsi-data read-only at /mnt/data and verifies training
    and validation files in place. No GPU. No local download. No writes.
    Returns a JSON-compatible summary dict.
    """
    results: Dict[str, Any] = {
        "gpu_requested": False,
        "cpu_requested": PREFLIGHT_CPU,
        "ram_requested_mb": PREFLIGHT_MEMORY_MIB,
        "timeout_seconds": PREFLIGHT_TIMEOUT_SECONDS,
        "volume_name": DATA_VOLUME_NAME,
        "mount_path": DATA_MOUNT,
        "mount_read_only": True,
        "training_path": DATASET_PATH,
        "validation_path": VAL_DATASET_PATH,
        "expected_train_size": EXPECTED_TRAIN_SIZE,
        "expected_train_sha256": EXPECTED_TRAIN_SHA256,
        "expected_val_size": EXPECTED_VAL_SIZE,
        "expected_val_sha256": EXPECTED_VAL_SHA256,
    }

    try:
        mnt_entries = sorted(os.listdir(DATA_MOUNT))
    except FileNotFoundError:
        mnt_entries = []
    results["mnt_data_contents"] = mnt_entries

    custom_dir = f"{DATA_MOUNT}/custom_75_25_english"
    try:
        custom_entries = sorted(os.listdir(custom_dir))
    except FileNotFoundError:
        custom_entries = []
    results["custom_75_25_english_contents"] = custom_entries

    train_exists = os.path.exists(DATASET_PATH)
    train_isfile = os.path.isfile(DATASET_PATH) if train_exists else False
    train_size = os.path.getsize(DATASET_PATH) if train_isfile else None
    train_sha256 = (
        _sha256_file_incremental(DATASET_PATH) if train_isfile else None
    )
    train_size_match = train_size == EXPECTED_TRAIN_SIZE
    train_hash_match = (
        train_sha256 == EXPECTED_TRAIN_SHA256 if train_sha256 else False
    )
    results["training_exists"] = train_exists
    results["training_isfile"] = train_isfile
    results["training_size"] = train_size
    results["training_size_match"] = train_size_match
    results["training_sha256"] = train_sha256
    results["training_hash_match"] = train_hash_match

    val_exists = os.path.exists(VAL_DATASET_PATH)
    val_isfile = os.path.isfile(VAL_DATASET_PATH) if val_exists else False
    val_size = os.path.getsize(VAL_DATASET_PATH) if val_isfile else None
    val_sha256 = (
        _sha256_file_incremental(VAL_DATASET_PATH) if val_isfile else None
    )
    val_size_match = val_size == EXPECTED_VAL_SIZE
    val_hash_match = (
        val_sha256 == EXPECTED_VAL_SHA256 if val_sha256 else False
    )
    results["validation_exists"] = val_exists
    results["validation_isfile"] = val_isfile
    results["validation_size"] = val_size
    results["validation_size_match"] = val_size_match
    results["validation_sha256"] = val_sha256
    results["validation_hash_match"] = val_hash_match

    mandatory_ok = (
        train_exists
        and train_isfile
        and train_size_match
        and train_hash_match
        and val_exists
        and val_isfile
        and val_size_match
        and val_hash_match
    )
    results["overall"] = "PASS" if mandatory_ok else "FAIL"
    return results


def preflight_validation() -> Dict[str, Any]:
    """Invoke remote CPU-only preflight before any GPU allocation.

    Execution order gate for training launch:
      1. Local Modal CLI already initialized (@app.local_entrypoint).
      2. Invoke run_dataset_preflight.remote() (CPU-only container).
      3. Parse the JSON-compatible result.
      4. Confirm both datasets pass all checks.
      5. Return the result so main() may then call train.remote().

    On failure or remote timeout: raise RuntimeError. Do not retry.
    Never downloads Volume files to the local machine.
    """
    print("=" * 60)
    print("Preflight validation (remote CPU-only, before GPU allocation)")
    print("=" * 60)
    print(f"Volume: {DATA_VOLUME_NAME} (read-only at {DATA_MOUNT})")
    print(
        f"CPU: {PREFLIGHT_CPU} cores, RAM: {PREFLIGHT_MEMORY_MIB} MiB, "
        f"Timeout: {PREFLIGHT_TIMEOUT_SECONDS}s"
    )
    print("GPU: none (preflight does not allocate RTX PRO 6000)")
    print(f"Training path: {DATASET_PATH}")
    print(f"Validation path: {VAL_DATASET_PATH}")
    print("=" * 60)

    try:
        result = run_dataset_preflight.remote()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Preflight failed: remote CPU dataset check did not complete "
            f"({type(exc).__name__}: {exc}). "
            "Aborting before GPU allocation. No automatic retry."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Preflight failed: expected a JSON-compatible dict, got "
            f"{type(result).__name__}. Aborting before GPU allocation."
        )

    # Ensure the payload is JSON-serializable before trusting it.
    try:
        serialized = json.dumps(result, indent=2)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Preflight failed: result is not JSON-serializable ({exc}). "
            "Aborting before GPU allocation."
        ) from exc

    print(serialized)

    failures: list[str] = []
    if result.get("overall") != "PASS":
        failures.append(f"overall={result.get('overall')!r}")
    if result.get("mount_path") != DATA_MOUNT:
        failures.append(
            f"mount_path={result.get('mount_path')!r} "
            f"(expected {DATA_MOUNT!r})"
        )
    if not result.get("training_exists"):
        failures.append(f"training missing at {DATASET_PATH}")
    if not result.get("training_isfile"):
        failures.append("training is not a regular file")
    if not result.get("training_size_match"):
        failures.append(
            f"training size={result.get('training_size')!r} "
            f"(expected {EXPECTED_TRAIN_SIZE})"
        )
    if not result.get("training_hash_match"):
        failures.append(
            f"training SHA-256={result.get('training_sha256')!r} "
            f"(expected {EXPECTED_TRAIN_SHA256})"
        )
    if not result.get("validation_exists"):
        failures.append(f"validation missing at {VAL_DATASET_PATH}")
    if not result.get("validation_isfile"):
        failures.append("validation is not a regular file")
    if not result.get("validation_size_match"):
        failures.append(
            f"validation size={result.get('validation_size')!r} "
            f"(expected {EXPECTED_VAL_SIZE})"
        )
    if not result.get("validation_hash_match"):
        failures.append(
            f"validation SHA-256={result.get('validation_sha256')!r} "
            f"(expected {EXPECTED_VAL_SHA256})"
        )
    if result.get("gpu_requested") is not False:
        failures.append(
            f"gpu_requested={result.get('gpu_requested')!r} "
            "(preflight must not request a GPU)"
        )

    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            f"Preflight failed: dataset checks did not pass ({detail}). "
            "Aborting before GPU allocation. No automatic retry."
        )

    print("[preflight] All checks passed.", flush=True)
    print("=" * 60, flush=True)
    return result


def validate_preflight_launch_result(result: Any) -> Dict[str, Any]:
    """Validate the CPU preflight payload before any training invocation.

    Raises RuntimeError when the result is missing, not a dict, not
    JSON-serializable, or does not report overall == \"PASS\".
    """
    if result is None:
        raise RuntimeError(
            "Preflight failed: missing result (got None). "
            "Aborting before GPU allocation. No automatic retry."
        )
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Preflight failed: expected a JSON-compatible dict, got "
            f"{type(result).__name__}. Aborting before GPU allocation."
        )
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Preflight failed: invalid JSON result ({exc}). "
            "Aborting before GPU allocation. No automatic retry."
        ) from exc
    if result.get("overall") != "PASS":
        raise RuntimeError(
            "Preflight failed: refusing to allocate RTX PRO 6000. "
            f"Result overall={result.get('overall')!r}."
        )
    return result


REQUIRED_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def validate_soup_config_target_modules() -> None:
    """Pre-allocation guard: verify the embedded Soup config declares an explicit,
    non-empty LoRA target_module list before any RTX PRO 6000 is requested.

    Decodes the base64-embedded Modal config (_CONFIG_B64) that is written to
    /root/soup.custom_75_25_english.modal.yaml inside the training container and
    consumed by `soup train`. Soup 0.73.3 converts "auto" -> None in
    soup_cli/trainer/sft.py and passes None to peft.get_peft_model, which raises
    "Please specify target_modules or target_parameters in peft_config". There is
    no automatic target-module resolution for model_type: gemma3 in Soup 0.73.3.

    The compatibility patch (apply_soup_compat_patch.py) only patches
    soup_cli.utils.gpu.model_size_from_name and never touches the YAML config, so
    validating the embedded config is sufficient to confirm the field survives the
    patch.

    Raises RuntimeError printing the exact incorrect configuration field if the
    list is missing, empty, None ("auto"), or does not contain all four Gemma 3
    attention targets.

    PyYAML is imported here (local-only) so remote Modal containers that import
    this module do not require the package. Install locally: PyYAML==6.0.3.
    """
    import yaml  # Local-only: not installed in Modal images

    config_bytes = base64.b64decode(_CONFIG_B64)
    cfg = yaml.safe_load(config_bytes)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            "Pre-allocation guard failed: embedded Soup config is not a mapping. "
            "Aborting before RTX PRO 6000 allocation."
        )
    training = cfg.get("training")
    if not isinstance(training, dict):
        raise RuntimeError(
            "Pre-allocation guard failed: embedded Soup config has no 'training' "
            "section. Aborting before RTX PRO 6000 allocation."
        )
    lora = training.get("lora")
    if not isinstance(lora, dict):
        raise RuntimeError(
            "Pre-allocation guard failed: embedded Soup config has no "
            "training.lora section. Aborting before RTX PRO 6000 allocation."
        )
    target_modules = lora.get("target_modules")
    if target_modules is None:
        raise RuntimeError(
            "Pre-allocation guard failed: training.lora.target_modules is None "
            "(Soup 0.73.3 converts 'auto' -> None before PEFT). "
            "Aborting before RTX PRO 6000 allocation. Set an explicit list, e.g. "
            "[q_proj, k_proj, v_proj, o_proj]."
        )
    if not isinstance(target_modules, list) or len(target_modules) == 0:
        raise RuntimeError(
            "Pre-allocation guard failed: training.lora.target_modules must be a "
            f"non-empty list, got {target_modules!r}. "
            "Aborting before RTX PRO 6000 allocation."
        )
    missing = [m for m in REQUIRED_LORA_TARGET_MODULES if m not in target_modules]
    if missing:
        raise RuntimeError(
            "Pre-allocation guard failed: training.lora.target_modules is missing "
            f"required Gemma 3 attention targets {missing}; got {target_modules!r}. "
            "Aborting before RTX PRO 6000 allocation."
        )
    print(
        "[guard] training.lora.target_modules = "
        f"{target_modules} (explicit, non-empty) — safe to allocate RTX PRO 6000",
        flush=True,
    )


def run_launch_sequence(
    *,
    preflight_callable: Callable[[], Any],
    train_callable: Callable[..., Any],
    max_runtime_seconds: int,
    stop_after_steps: Optional[int],
) -> Any:
    """Orchestrate CPU preflight then a single training invocation.

    This pure control-flow helper is unit-tested with mocks. It must not
    return successfully after a PASS without calling ``train_callable``
    exactly once. Detach does not auto-start training — only this call does.
    """
    print("[main] Entrypoint started", flush=True)
    print("[main] Running CPU preflight", flush=True)
    raw_result = preflight_callable()
    print("[main] CPU preflight returned", flush=True)

    try:
        preflight_result = validate_preflight_launch_result(raw_result)
    except RuntimeError:
        overall = (
            raw_result.get("overall")
            if isinstance(raw_result, dict)
            else "MISSING_OR_INVALID"
        )
        print(f"[main] Preflight result: {overall}", flush=True)
        raise

    print("[main] Preflight result: PASS", flush=True)
    print("Preflight passed. Invoking training function now.", flush=True)

    # Pre-allocation configuration guard: verify the embedded Soup config
    # declares an explicit LoRA target_module list before any GPU is requested.
    # Raises and aborts before train.remote() (RTX PRO 6000) if invalid.
    validate_soup_config_target_modules()

    print("[main] About to invoke train.remote()", flush=True)
    train_result = train_callable(
        max_runtime_seconds=max_runtime_seconds,
        stop_after_steps=stop_after_steps,
    )
    print("[main] Training function invocation submitted", flush=True)
    print(f"[main] Training call result/reference: {train_result!r}", flush=True)
    return train_result


def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Validate checkpoint files and ensure trainer_state matches its directory."""
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_dir.name)
    if not checkpoint_dir.is_dir() or match is None:
        return False

    expected_step = int(match.group(1))
    for fname in CHECKPOINT_REQUIRED_FILES:
        file_path = checkpoint_dir / fname
        try:
            if not file_path.is_file() or file_path.stat().st_size <= 0:
                return False
        except OSError:
            return False

    try:
        state = json.loads(
            (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    return state.get("global_step") == expected_step


def find_completed_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Find all completed checkpoints (have trainer_state.json).

    Returns list of (step, path) tuples sorted by step.
    """
    if not output_dir.exists():
        return []

    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[-1])
            except (ValueError, IndexError):
                continue
            if checkpoint_is_complete(d):
                checkpoints.append((step, d))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def find_latest_checkpoint_step(output_dir: Path) -> Optional[int]:
    """Find the latest completed checkpoint step.

    Returns None if no completed checkpoints exist.
    """
    completed = find_completed_checkpoints(output_dir)
    if not completed:
        return None
    return completed[-1][0]


def commit_checkpoint_volume(committed_steps_arg: set[int]) -> set[int]:
    """Commit the checkpoint Volume for newly completed checkpoints only.

    Scans for checkpoint directories with trainer_state.json that haven't
    been committed yet. Commits the Volume once if any new checkpoints exist.
    Returns the updated set of committed step numbers.
    """
    with commit_lock:
        newly_committed = set()
        completed = find_completed_checkpoints(Path(OUTPUT_DIR))
        for step, _ in completed:
            if step not in committed_steps_arg:
                newly_committed.add(step)

        if newly_committed:
            try:
                checkpoint_volume.commit()
                print(
                    f"Checkpoint Volume committed for steps: "
                    f"{sorted(newly_committed)}"
                )
                # Only update tracking and pointer on successful commit.
                # On failure, committed_steps_arg is unchanged so the
                # previous valid checkpoint remains recoverable.
                latest = max(newly_committed)
                update_latest_valid_checkpoint(Path(OUTPUT_DIR), latest)
                # Commit the Volume again so the latest-valid-checkpoint
                # pointer file itself is durable on Modal storage.
                checkpoint_volume.commit()
                committed_steps_arg = committed_steps_arg | newly_committed
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Checkpoint Volume commit failed: {exc}")

        return committed_steps_arg


def commit_hf_cache_volume() -> None:
    """Commit the HF cache Volume after model download completes.

    Called after training finishes (model is cached by then).
    """
    with commit_lock:
        try:
            hf_cache_volume.commit()
            print("HF cache Volume committed.")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] HF cache Volume commit failed: {exc}")


def commit_dataset_volume() -> None:
    """Commit the dataset Volume after upload and checksum verification.

    Called once at startup after dataset validation succeeds.
    """
    with commit_lock:
        try:
            data_volume.commit()
            print("Dataset Volume committed.")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Dataset Volume commit failed: {exc}")


def update_latest_valid_checkpoint(output_dir: Path, step: int) -> None:
    """Atomically update the latest-valid-checkpoint pointer.

    Writes to a temp file then atomically replaces the pointer. If the
    process crashes during write, the previous pointer is preserved.
    """
    pointer_path = Path(output_dir) / "latest-valid-checkpoint"
    temp_path = pointer_path.with_suffix(".tmp")
    temp_path.write_text(str(step), encoding="utf-8")
    temp_path.replace(pointer_path)


def generate_manifest() -> None:
    """Generate a checkpoint manifest with metadata and SHA256 checksums.

    Written to OUTPUT_DIR/checkpoint_manifest.json AFTER all checkpoints
    have been committed. Includes versions, training config fingerprint,
    dataset fingerprint, checkpoint file list, and checksums.
    """
    import importlib.metadata

    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        return

    # Collect dependency versions
    versions = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "soup-cli"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"

    # Collect checkpoint information (only completed checkpoints)
    checkpoints = []
    for step, d in find_completed_checkpoints(output_dir):
        files = []
        for f in sorted(d.iterdir()):
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size": f.stat().st_size,
                        "sha256": compute_sha256(f),
                    }
                )

        checkpoints.append(
            {
                "step": step,
                "path": d.name,
                "files": files,
            }
        )

    # Dataset fingerprint
    dataset_sha256 = ""
    dataset_path = Path(DATASET_PATH)
    if dataset_path.exists():
        dataset_sha256 = compute_sha256(dataset_path)

    # Config fingerprint (hash of the exact embedded Modal training config).
    config_sha256 = ""
    config_path = Path("/root/soup.custom_75_25_english.modal.yaml")
    if config_path.exists():
        config_sha256 = compute_sha256(config_path)

    manifest = {
        "project": "gemma-tounsi-1.0-27b-custom-75-25-english",
        "base_model": "google/gemma-3-27b-it",
        "base_model_revision": get_model_revision(),
        "soup_version": "0.73.3",
        "python_version": sys.version,
        "dependencies": versions,
        "training_config_fingerprint": config_sha256,
        "dataset_fingerprint": dataset_sha256,
        "checkpoints": checkpoints,
    }

    manifest_path = Path(CHECKPOINT_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Checkpoint manifest written: {manifest_path}")


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=MODAL_FUNCTION_TIMEOUT_SECONDS,
    secrets=[hf_secret],
    volumes={
        DATA_MOUNT: data_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def train(
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
    stop_after_steps: Optional[int] = STOP_AFTER_STEPS,
) -> None:
    """Run Soup SFT training with checkpoint persistence and resume support.

    BLOCKER 6: Explicit checkpoint-path resumption; no fresh-start fallback.
    BLOCKER 7: Runtime timeout and soft-stop protection (23h margin).
    BLOCKER 4: Per-checkpoint Volume commits after every completed checkpoint.

    Commit safety:
      * No periodic background commits — avoids committing partial checkpoints.
      * Commits happen after EVERY completed checkpoint (trainer_state.json
        present), not just after the subprocess exits.
      * Only checkpoints with trainer_state.json (complete) are committed.
      * A threading lock prevents concurrent commits.
      * Failed commits preserve the previous valid checkpoint pointer.
    """
    global committed_steps

    # Internal runtime clock — starts when train() begins, before any work.
    training_start_time = time.monotonic()

    # Set HF cache environment variables (BLOCKER 5)
    os.environ["HF_HOME"] = HF_CACHE_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_CACHE_MOUNT}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_MOUNT}/transformers"

    # Write embedded Modal-specific config
    config_path = "/root/soup.custom_75_25_english.modal.yaml"
    config_bytes = base64.b64decode(_CONFIG_B64)
    Path(config_path).write_bytes(config_bytes)

    # BLOCKER 3: Validate dataset exists and (optionally) SHA256
    dataset_sha256 = validate_dataset()

    # Commit dataset Volume after successful validation (BLOCKER 4)
    commit_dataset_volume()

    # BLOCKER 2: Validate HF_TOKEN is available from the Secret
    validate_hf_token()

    # BLOCKER 7: Check if STOP_AFTER_STEPS already reached by existing checkpoint
    output_dir = Path(OUTPUT_DIR)
    if stop_after_steps is not None:
        latest_step = find_latest_checkpoint_step(output_dir)
        if latest_step is not None and latest_step >= stop_after_steps:
            print(
                f"Checkpoint at step {latest_step} >= STOP_AFTER_STEPS="
                f"{stop_after_steps}. Training already complete."
            )
            return

    try:
        # Never cold-start: resume the newest complete checkpoint, but never
        # accept a recovery point older than the verified Step 900 checkpoint.
        latest_step = find_latest_checkpoint_step(output_dir)
        if latest_step is None:
            raise RuntimeError(
                "No complete checkpoint found. Refusing to start training from scratch."
            )
        if latest_step < MIN_RESUME_STEP:
            raise RuntimeError(
                f"Latest complete checkpoint is {latest_step}, below the required "
                f"recovery step {MIN_RESUME_STEP}. Refusing to start."
            )

        resume_checkpoint = output_dir / f"checkpoint-{latest_step}"
        if not checkpoint_is_complete(resume_checkpoint):
            raise RuntimeError(
                f"Selected checkpoint is incomplete: {resume_checkpoint}"
            )

        trainer_state_path = resume_checkpoint / "trainer_state.json"
        try:
            trainer_state = json.loads(
                trainer_state_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read trainer state from {trainer_state_path}: {exc}"
            ) from exc

        recorded_step = trainer_state.get("global_step")
        if recorded_step != latest_step:
            raise RuntimeError(
                f"Checkpoint directory says step {latest_step}, but trainer_state.json "
                f"records global_step={recorded_step!r}. Refusing to resume."
            )

        print(
            f"[resume] Resuming explicitly from {resume_checkpoint} "
            f"(global_step={recorded_step}).",
            flush=True,
        )
        resume_args = ["--resume", str(resume_checkpoint)]

        # Remember the latest checkpoint that existed before this run.
        # At the soft runtime limit, wait for one newer checkpoint to be
        # fully written and committed before requesting process termination.
        initial_checkpoint_step = latest_step

        # Build soup train command (BLOCKER 6)
        cmd = [
            "soup",
            "train",
            "--config",
            config_path,
            "--yes",
        ]
        cmd.extend(resume_args)

        print(f"Starting training: {' '.join(cmd)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Max runtime: {max_runtime_seconds}s")
        if stop_after_steps:
            print(f"Stop after steps: {stop_after_steps}")

        # The internal soft-stop must leave time for the parent function to
        # commit Volumes and write the manifest before Modal's hard timeout.
        forced_stop_elapsed = (
            MODAL_FUNCTION_TIMEOUT_SECONDS - FINALIZATION_RESERVE_SECONDS
        )
        if max_runtime_seconds <= 0 or max_runtime_seconds >= forced_stop_elapsed:
            raise ValueError(
                "max_runtime_seconds must be positive and leave the configured "
                "finalization reserve before Modal's hard timeout."
            )

        process = subprocess.Popen(cmd)
        termination_requested = False
        termination_reason: Optional[str] = None

        def request_sigterm(reason: str) -> None:
            """Request process termination after checkpoint durability when possible.

            Transformers 4.x, which is the range allowed by Soup 0.73.3,
            does not expose the newer JIT SIGTERM checkpoint option. This
            SIGTERM is therefore not assumed to trigger a Trainer save.
            The normal stop path sends it only after a newer checkpoint has
            completed and been committed. The final-deadline path preserves
            the last committed checkpoint and may discard newer in-flight work.
            """
            nonlocal termination_requested, termination_reason
            termination_requested = True
            termination_reason = reason
            print(f"[shutdown] Sending SIGTERM to Soup: {reason}")
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                # The process exited between poll() and signal delivery.
                pass

        soft_stop_requested = False
        budget_stop_requested = False

        # Monitor the subprocess and commit every complete checkpoint.
        while process.poll() is None:
            elapsed = time.monotonic() - training_start_time

            latest_step_now = find_latest_checkpoint_step(output_dir)
            committed_steps = commit_checkpoint_volume(committed_steps)

            # Budget-aware early stop: when estimated training spend approaches
            # the $24.00 ceiling, request a graceful stop at the next committed
            # checkpoint. This preserves the latest valid checkpoint and leaves
            # room for the $2.00 finalization phase. Never auto-restart.
            # Uses the ALL-IN rate (GPU + CPU + RAM), not GPU-only. A GPU-only
            # timer is not a guaranteed total-cost cap.
            estimated_cost = elapsed * ALL_IN_COST_PER_SECOND
            if (
                not budget_stop_requested
                and elapsed >= BUDGET_STOP_SECONDS
            ):
                budget_stop_requested = True
                print(
                    f"[budget] Estimated all-in training cost ${estimated_cost:.2f} "
                    f"(rate ${ALL_IN_COST_PER_SECOND:.6f}/s = GPU+CPU+RAM) >= "
                    f"${TRAINING_BUDGET_USD - BUDGET_SAFETY_MARGIN_USD:.2f} "
                    f"ceiling (budget ${TRAINING_BUDGET_USD:.2f}, minus "
                    f"${IMAGE_BUILD_BUFFER_USD:.2f} image-build buffer). "
                    "Requesting graceful stop at next committed checkpoint."
                )

            # Explicit step-limited runs stop only after the target checkpoint
            # has passed validation and has been committed to the Volume.
            if stop_after_steps is not None:
                if (
                    latest_step_now is not None
                    and latest_step_now >= stop_after_steps
                    and latest_step_now in committed_steps
                ):
                    request_sigterm(
                        f"STOP_AFTER_STEPS={stop_after_steps} reached; "
                        f"checkpoint {latest_step_now} is committed"
                    )
                    break

            if elapsed >= max_runtime_seconds and not soft_stop_requested:
                soft_stop_requested = True
                print(
                    f"[shutdown] {max_runtime_seconds}s soft runtime reached. "
                    "Waiting for a newer complete checkpoint to be committed."
                )

            if soft_stop_requested:
                has_new_committed_checkpoint = (
                    latest_step_now is not None
                    and (
                        initial_checkpoint_step is None
                        or latest_step_now > initial_checkpoint_step
                    )
                    and latest_step_now in committed_steps
                )
                if has_new_committed_checkpoint:
                    request_sigterm(
                        f"new checkpoint {latest_step_now} is complete and committed"
                    )
                    break

            if budget_stop_requested:
                has_new_committed_checkpoint = (
                    latest_step_now is not None
                    and (
                        initial_checkpoint_step is None
                        or latest_step_now > initial_checkpoint_step
                    )
                    and latest_step_now in committed_steps
                )
                if has_new_committed_checkpoint:
                    request_sigterm(
                        f"training budget nearly exhausted (est. "
                        f"${estimated_cost:.2f} of ${TRAINING_BUDGET_USD:.2f}); "
                        f"checkpoint {latest_step_now} is committed"
                    )
                    break

            # Last-resort stop before Modal's hard timeout. This may interrupt
            # the current step, but the last fully committed checkpoint stays
            # the recovery point; incomplete checkpoint directories are ignored.
            if elapsed >= forced_stop_elapsed:
                request_sigterm(
                    "finalization reserve reached; preserving the latest "
                    "fully committed checkpoint"
                )
                break

            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)

        if termination_requested:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(
                    "[warn] Soup did not exit after SIGTERM within 60 seconds; "
                    "sending SIGKILL. The last committed checkpoint remains "
                    "the recovery point."
                )
                process.kill()
                process.wait()
        else:
            process.wait()

        expected_signal_exit = termination_requested and process.returncode in (
            -signal.SIGTERM,
            -signal.SIGKILL,
        )
        if process.returncode != 0 and not expected_signal_exit:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        if termination_requested:
            last_committed_step = max(committed_steps) if committed_steps else None
            print(
                f"Training process stopped ({termination_reason}). "
                f"Last committed checkpoint step: {last_committed_step}."
            )
        else:
            print(f"Training completed with return code {process.returncode}")

    finally:
        # Final per-checkpoint commit for any checkpoints saved after the
        # last poll cycle. HF Trainer has finished writing all checkpoints
        # by the time the subprocess exits.
        committed_steps = commit_checkpoint_volume(committed_steps)

        # Commit HF cache Volume after model download completes (BLOCKER 5)
        commit_hf_cache_volume()

        # Generate checkpoint manifest AFTER successful commits (BLOCKER 6)
        generate_manifest()


def _invoke_train_remote(
    *,
    max_runtime_seconds: int,
    stop_after_steps: Optional[int],
) -> Any:
    """Submit training exactly once via ``train.remote()`` and return its result.

    ``modal run --detach`` does **not** auto-start training after preflight.
    Training runs only when this function executes. Detached mode only keeps
    the last triggered Modal function alive after local disconnect — so this
    call must happen explicitly after a PASS preflight.
    """
    return train.remote(
        max_runtime_seconds=max_runtime_seconds,
        stop_after_steps=stop_after_steps,
    )


@app.local_entrypoint()
def main(
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
    stop_after_steps: Optional[int] = None,
) -> None:
    """Entry point for `modal run --detach modal_train_custom_75_25_english.py`.

    Args:
        max_runtime_seconds: Maximum runtime per session (default 82800s = 23h).
        stop_after_steps: Stop training after this many completed steps.
    """
    print("=" * 60)
    print("Gemma Tounsi 1.0 (27B) — Modal QLoRA Training (custom_75_25_english)")
    print("=" * 60)
    print(f"App: gemma-tounsi-train")
    print("GPU: RTX-PRO-6000 (NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM) — exactly one GPU")
    print(f"Soup: 0.73.3 (patched for 27B model-size detection)")
    print(f"Dataset: {DATASET_PATH} (Volume: {DATA_VOLUME_NAME})")
    print(f"Output: {OUTPUT_DIR} (Volume: {CHECKPOINT_VOLUME_NAME})")
    print(f"HF Cache: {HF_CACHE_MOUNT} (Volume: {HF_CACHE_VOLUME_NAME})")
    print(f"HF Secret: gemma-hf (provides HF_TOKEN)")
    print(f"Max runtime: {max_runtime_seconds}s")
    if stop_after_steps:
        print(f"Stop after steps: {stop_after_steps}")
    print(f"Training budget: ${TRAINING_BUDGET_USD:.2f} (all-in training ceiling)")
    print(
        f"Budget-aware stop at ~${TRAINING_BUDGET_USD - BUDGET_SAFETY_MARGIN_USD - IMAGE_BUILD_BUFFER_USD:.2f} "
        f"estimated all-in cost ({BUDGET_STOP_SECONDS:.0f}s / {BUDGET_STOP_SECONDS/3600:.1f}h, "
        f"all-in rate ${ALL_IN_COST_PER_SECOND:.6f}/s)"
    )
    print(
        f"Note: GPU-only rate ${GPU_COST_PER_SECOND:.6f}/s; all-in rate includes "
        f"CPU (${CPU_COST_PER_SECOND:.6f}/s) + RAM (${RAM_COST_PER_SECOND:.6f}/s). "
        f"Image build (${IMAGE_BUILD_BUFFER_USD:.2f} buffer) billed separately."
    )
    print("=" * 60)

    # Strict launch gate (explicit — detach never auto-invokes train):
    #   1. Local Modal CLI / app entrypoint already initialized.
    #   2-4. Remote CPU-only preflight must PASS (raises on failure/timeout).
    #   5. Only then invoke train.remote() exactly once (RTX PRO 6000).
    # Datasets are verified inside Modal at /mnt/data — never downloaded locally.
    run_launch_sequence(
        preflight_callable=preflight_validation,
        train_callable=_invoke_train_remote,
        max_runtime_seconds=max_runtime_seconds,
        stop_after_steps=stop_after_steps,
    )

    print("=" * 60)
    print("Training session complete.")
    print(f"Download checkpoints: modal volume get {CHECKPOINT_VOLUME_NAME} .")
    print(f"Resume training: modal run --detach modal_train_custom_75_25_english.py")
    print("=" * 60)
