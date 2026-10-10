"""Run the Soup 0.73.3 dataset toolchain over the MSA candidate pools.

WHY THIS EXISTS
---------------
``docs/data/msa_audit.md`` reports Soup numbers (validate 10,000/10,000; dedup
removed 35 / 2,128; langdetect "unknown") but those numbers live only as PROSE
in the document. ``data/processed/msa/audit_results.json`` -- the one
machine-readable artifact -- records ``near_duplicates: 0`` and
``near_dup_pairs: null``, because near-duplicate analysis was delegated to Soup
and never written back. The Soup half of the audit is therefore not
reproducible: nobody can re-derive it without retyping numbers by hand.

This driver closes that gap. It runs the Soup data commands over the MSA pools,
captures every invocation VERBATIM, and writes a machine-readable report so the
audit's Soup numbers become checkable artifacts instead of prose.

DESIGN RULES (deliberate, and the reason this is a driver and not a one-liner)
-----------------------------------------------------------------------------
1. NO INVENTED FLAGS. Only the ``soup data`` flags this repository has already
   verified against 0.73.3 are emitted:

     soup data validate PATH --format {alpaca,sharegpt,chatml,dpo,kto,plaintext}
                                                   (src/data/export.py,
                                                    src/data/validate.py,
                                                    scripts/prepare_data.sh)
     soup data stats PATH                          (src/data/stats.py)
     soup data inspect PATH                        (src/data/stats.py)
     soup data dedup PATH --output OUT --threshold T [--field F] [--semantic]
                                                   (src/data/dedupe.py)
     soup data langdetect -i IN -o OUT             (docs/data/msa_audit.md)

   Commands whose flags are NOT verified here (``doctor``, ``decontaminate``,
   ``convert``, ``merge``, ``split``) are NEVER guessed at. Instead the run
   captures ``soup data --help`` so the real surface is recorded as evidence,
   and they can be added once read from that output.

2. CONTAINER-FIRST. Soup is an external dependency that only exists inside the
   pinned image (``ghcr.io/makazhanalpamys/soup:0.73.3``). If ``soup`` is not on
   PATH the driver prints the exact ``docker compose run`` command and exits
   non-zero -- it never pretends to have run.

3. RAW OUTPUT IS THE SOURCE OF TRUTH. Every invocation's stdout/stderr is kept
   verbatim under ``data/processed/msa/soup/logs/``. Parsed numbers are
   best-effort and each carries ``"parsed": true|false``; an unparsed value is
   reported as ``null``, NEVER as a plausible-looking guess.

4. NON-DESTRUCTIVE. The September 4 artifacts referenced by the audit document
   (``data/processed/msa/*_deduped.jsonl``, ``*_langdetect.jsonl``) are left
   untouched. This run writes into ``data/processed/msa/soup/`` so old and new
   results can be compared rather than silently replaced.

5. DETERMINISTIC PREFLIGHT. CIDAR ships as Parquet, which is not one of Soup's
   wire formats, so it must be materialised as alpaca JSONL first. That
   conversion is explicit, stable-ordered, and recorded with a SHA-256 so the
   exact bytes Soup consumed are identifiable later.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
- It does not download anything. Inputs are the immutable files already in
  ``data/raw/`` (see ``scripts/acquire_msa_candidate_data.py``).
- It does not train, and it does not touch a GPU.
- It does not run a contamination check against the eval sets. That is an open
  TODO in ``data/manifests/msa.yaml``, and it is IMPOSSIBLE today:
  ``data/manifests/eval.yaml`` is a skeleton with ``sources: []`` for all three
  tracks, so there is nothing to decontaminate against. The driver records that
  as a blocker rather than running a check that would trivially "pass".
- It does not touch PALM or ArSyra: both are blocked (gated + licence), see
  ``data/manifests/msa.yaml`` -> ``blocked_candidates``.

USAGE
-----
    # Inside the Soup container (the real run):
    docker compose run --rm soup-cpu python scripts/soup_analyze_msa.py

    # Same thing, letting the script re-exec itself into the container:
    python scripts/soup_analyze_msa.py --in-docker

    # Offline: materialise Soup inputs + print the exact plan, run nothing:
    python scripts/soup_analyze_msa.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess  # noqa: S404 - invokes the pinned Soup CLI, never shell=True
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: Where this run's artifacts go. Deliberately a SUBDIRECTORY so the September 4
#: artifacts cited by docs/data/msa_audit.md are not overwritten.
OUT_DIR = ROOT / "data" / "processed" / "msa" / "soup"
LOG_DIR = OUT_DIR / "logs"

#: Soup wire format used for both pools. `alpaca` is the instruction/input/output
#: shape both corpora already have; it is one of the formats
#: `soup data validate --format` accepts in 0.73.3 (see src/data/export.py).
SOUP_FORMAT = "alpaca"

#: Same threshold as the retention reports and the existing MSA audit, so the new
#: numbers are directly comparable to the ones already in the document.
DEDUP_THRESHOLD = 0.85

#: Strict alpaca columns. Anything else (CIDAR's `index`, Arabic QA's `source` /
#: `language`) is stripped out of the Soup input and preserved in a sidecar
#: id-map instead -- the same provenance pattern src/data/palm_msa.py uses.
ALPACA_FIELDS = ("instruction", "input", "output")

#: Subcommands whose ``--help`` is captured by ``--probe-help``. This is how a
#: flag is allowed to enter the plan: it is READ from the installed Soup, never
#: guessed. The list is the subset of ``soup data`` (44 subcommands in 0.73.3)
#: that could plausibly say something about a static, offline Arabic
#: instruction corpus. Deliberately excluded: anything needing network or API
#: keys (search, preview, download, push, augment, generate, forge, gen-magpie,
#: best-of-n, evolve, persona-mix, active-sample, from-traces), anything
#: preference-pair-only (lint, review), and registry/demo plumbing.
HELP_PROBE_COMMANDS: Tuple[str, ...] = (
    "validate",
    "stats",
    "inspect",
    "dedup",
    "langdetect",
    "decontaminate",
    "topics",
    "score",
    "pii",
    "toxicity",
    "educational",
    "doctor",
    "canary",
    "split",
    "filter",
    "brain-rot",
)



# --------------------------------------------------------------------------- #
# Pool definitions
# --------------------------------------------------------------------------- #
# Paths mirror `local_path` in data/manifests/msa.yaml. PALM and ArSyra are
# absent on purpose: they are under `blocked_candidates`, not `sources`.


class Pool:
    """One MSA candidate pool and how to read it.

    Attributes:
        pool_id: manifest source id (``data/manifests/msa.yaml`` -> ``sources``).
        raw_path: immutable input file under ``data/raw/``.
        reader: ``"parquet"`` or ``"jsonl"``.
    """

    def __init__(self, pool_id: str, raw_path: Path, reader: str) -> None:
        self.pool_id = pool_id
        self.raw_path = raw_path
        self.reader = reader

    @property
    def soup_input(self) -> Path:
        """Alpaca JSONL materialised for Soup to consume."""
        return OUT_DIR / f"{self.pool_id}_alpaca.jsonl"

    @property
    def idmap(self) -> Path:
        """Sidecar provenance map (row -> original identifiers)."""
        return OUT_DIR / f"{self.pool_id}_idmap.jsonl"

    @property
    def deduped(self) -> Path:
        """Where ``soup data dedup`` writes its output for this pool."""
        return OUT_DIR / f"{self.pool_id}_deduped.jsonl"

    @property
    def langdetect(self) -> Path:
        """Where ``soup data langdetect`` writes its output for this pool."""
        return OUT_DIR / f"{self.pool_id}_langdetect.jsonl"


POOLS: Tuple[Pool, ...] = (
    Pool(
        "cidar",
        ROOT / "data" / "raw" / "arbml__CIDAR" / "data"
        / "train-00000-of-00001-b2881e1b9f14c3b1.parquet",
        "parquet",
    ),
    Pool(
        "arabic_qa",
        ROOT / "data" / "raw" / "bobez999__arabic-qa-dataset-sigir2024" / "data"
        / "arabic_qa_10k_sample.jsonl",
        "jsonl",
    ),
)


# --------------------------------------------------------------------------- #
# Preflight: materialise Soup-readable alpaca JSONL
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a Parquet file into row dicts, preserving file order."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pyarrow is required to read CIDAR's Parquet file. It ships with "
            "the Soup image (datasets depends on it); on a bare host install "
            "it with 'pip install pyarrow'."
        ) from exc

    table = pq.read_table(path)
    columns = table.to_pydict()
    names = list(columns)
    length = len(columns[names[0]]) if names else 0
    return [{name: columns[name][i] for name in names} for i in range(length)]


def _read_jsonl_rows(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read a JSONL file into row dicts. Returns ``(rows, bad_json_lines)``."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(record, dict):
                rows.append(record)
            else:
                bad += 1
    return rows, bad


def _as_text(value: Any) -> str:
    """Coerce a field to a string without inventing content."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def prepare_pool(pool: Pool) -> Dict[str, Any]:
    """Materialise ``pool`` as strict-alpaca JSONL plus a provenance id-map.

    The Soup input contains ONLY ``instruction``/``input``/``output`` so that
    ``soup data validate --format alpaca`` judges the data, not our extra
    columns. Every other field is preserved row-for-row in the id-map, so no
    provenance is lost.

    Also measures, LOCALLY (not with Soup), the structural facts the audit
    relies on: empty-field rows and unique-passage counts. These are labelled as
    local in the report and never attributed to Soup.

    Returns:
        A summary dict for the run report.
    """
    if not pool.raw_path.exists():
        raise FileNotFoundError(
            f"{pool.pool_id}: raw input not found at {pool.raw_path}. "
            f"Run scripts/acquire_msa_candidate_data.py first."
        )

    bad_json = 0
    if pool.reader == "parquet":
        rows = _read_parquet_rows(pool.raw_path)
    else:
        rows, bad_json = _read_jsonl_rows(pool.raw_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    empty_io = 0
    passage_hashes: Dict[str, int] = {}
    written = 0

    with pool.soup_input.open("w", encoding="utf-8", newline="\n") as out, \
            pool.idmap.open("w", encoding="utf-8", newline="\n") as imap:
        for row_number, row in enumerate(rows):
            instruction = _as_text(row.get("instruction"))
            input_text = _as_text(row.get("input"))
            output = _as_text(row.get("output"))

            if not instruction.strip() or not output.strip():
                empty_io += 1

            # Rows are passed to Soup AS-IS, including the empty-output row.
            # Dropping it here would hide it from `soup data validate`, and the
            # whole point of validation is to let Soup find it.
            out.write(
                json.dumps(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            provenance: Dict[str, Any] = {"row": row_number}
            for name, value in row.items():
                if name not in ALPACA_FIELDS:
                    provenance[name] = value
            if input_text.strip():
                # Passage identity, so the "<= 2 questions per passage" cap in
                # configs/data/msa.yaml is reproducible rather than asserted.
                passage_id = hashlib.sha256(
                    input_text.strip().encode("utf-8")
                ).hexdigest()
                provenance["passage_sha256"] = passage_id
                passage_hashes[passage_id] = passage_hashes.get(passage_id, 0) + 1
            imap.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            written += 1

    summary: Dict[str, Any] = {
        "pool_id": pool.pool_id,
        "raw_path": pool.raw_path.relative_to(ROOT).as_posix(),
        "raw_sha256": _sha256(pool.raw_path),
        "reader": pool.reader,
        "rows_read": len(rows),
        "rows_written": written,
        "bad_json_lines": bad_json,
        "soup_input": pool.soup_input.relative_to(ROOT).as_posix(),
        "soup_input_sha256": _sha256(pool.soup_input),
        "idmap": pool.idmap.relative_to(ROOT).as_posix(),
        # --- local measurements, NOT Soup output -------------------------------
        "local_measurements": {
            "empty_instruction_or_output": empty_io,
            "rows_with_passage": sum(passage_hashes.values()),
            "unique_passages": len(passage_hashes) or None,
            "max_questions_per_passage": max(passage_hashes.values())
            if passage_hashes
            else None,
            "rows_sharing_a_passage": sum(
                count for count in passage_hashes.values() if count > 1
            )
            or None,
        },
    }
    return summary


# --------------------------------------------------------------------------- #
# Soup command plan
# --------------------------------------------------------------------------- #


class SoupStep:
    """One Soup invocation, with the reason it is in the plan.

    Attributes:
        name: stable slug used for the log filename and report key.
        argv: full command line, starting with the Soup executable.
        why: short justification, so the plan is self-documenting.
        writes: path Soup is expected to create, if any.
    """

    def __init__(
        self,
        name: str,
        argv: Sequence[str],
        why: str,
        writes: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.argv = tuple(argv)
        self.why = why
        self.writes = writes

    def as_str(self) -> str:
        """Return the command as a copy-pasteable string."""
        return " ".join(self.argv)


def build_plan(soup: str = "soup") -> List[SoupStep]:
    """Build the full ordered list of Soup invocations.

    Only verified 0.73.3 flags are used (see the module docstring). Relative
    paths are emitted so the plan reads identically on the host and at
    ``/workspace`` inside the container.
    """
    def rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    steps: List[SoupStep] = [
        SoupStep(
            "soup_version",
            [soup, "version"],
            "Pin the exact Soup build every number below came from. "
            "`soup --version` does not exist in 0.73.3; the subcommand does.",
        ),
        SoupStep(
            "soup_data_help",
            [soup, "data", "--help"],
            "Record the REAL `soup data` subcommand surface as evidence, so "
            "commands this repo has not yet verified (doctor, decontaminate, "
            "convert, merge, split) can be added from observed help text "
            "instead of being guessed.",
        ),
    ]

    for pool in POOLS:
        pid = pool.pool_id
        steps += [
            SoupStep(
                f"{pid}__validate",
                [soup, "data", "validate", rel(pool.soup_input),
                 "--format", SOUP_FORMAT],
                f"Format validation of {pid}: does Soup accept every row as "
                f"{SOUP_FORMAT}? This is what finds malformed rows.",
            ),
            SoupStep(
                f"{pid}__inspect",
                [soup, "data", "inspect", rel(pool.soup_input)],
                f"Sample rows of {pid} straight from Soup -- confirms Soup "
                f"parses the Arabic text as expected, not as mojibake.",
            ),
            SoupStep(
                f"{pid}__stats",
                [soup, "data", "stats", rel(pool.soup_input)],
                f"Length/token statistics for {pid} BEFORE dedup.",
            ),
            SoupStep(
                f"{pid}__dedup",
                [soup, "data", "dedup", rel(pool.soup_input),
                 "--output", rel(pool.deduped),
                 "--threshold", str(DEDUP_THRESHOLD)],
                f"MinHash Jaccard near-duplicate removal on {pid} at "
                f"{DEDUP_THRESHOLD} -- the same threshold as the retention "
                f"reports, so the numbers stay comparable. This is the step "
                f"whose result audit_results.json is currently missing.",
                writes=pool.deduped,
            ),
            SoupStep(
                f"{pid}__stats_deduped",
                [soup, "data", "stats", rel(pool.deduped)],
                f"Statistics for {pid} AFTER dedup. The existing audit only "
                f"reported pre-dedup stats, so the post-dedup distribution "
                f"(what selection actually draws from) was never measured.",
            ),
            SoupStep(
                f"{pid}__langdetect",
                [soup, "data", "langdetect",
                 "-i", rel(pool.soup_input),
                 "-o", rel(pool.langdetect)],
                f"Language tagging for {pid}. Expected to be of LIMITED value: "
                f"the audit found this heuristic is Latin-script-centric and "
                f"tags Arabic-script rows 'unknown'. Re-run to confirm that "
                f"limitation still holds in 0.73.3 rather than assuming it.",
                writes=pool.langdetect,
            ),
        ]
    return steps


# --------------------------------------------------------------------------- #
# Execution + capture
# --------------------------------------------------------------------------- #

#: Tolerant extractors. Soup's exact stdout wording is NOT assumed: each pattern
#: is tried, and a miss is reported as null with "parsed": false. Nothing here
#: ever invents a number.
_NUMBER_HINTS: Tuple[Tuple[str, str], ...] = (
    ("valid", r"(?i)\bvalid\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
    ("invalid", r"(?i)\binvalid\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
    ("errors", r"(?i)\berrors?\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
    ("total_rows", r"(?i)\b(?:total|examples?|rows?|records?|samples?)\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
    ("removed", r"(?i)\b(?:removed|dropped|duplicates?)\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
    ("kept", r"(?i)\b(?:kept|remaining|unique|after)\b[^0-9\n]{0,40}([0-9][0-9,]*)"),
)


def _extract_numbers(text: str) -> Dict[str, Any]:
    """Best-effort numeric extraction from Soup stdout.

    Every key carries its own ``parsed`` flag. This is intentionally weak: the
    authoritative record is the raw log. The parsed view exists only to make the
    report convenient, and must never be mistaken for ground truth.
    """
    import re

    found: Dict[str, Any] = {}
    for key, pattern in _NUMBER_HINTS:
        match = re.search(pattern, text)
        if match:
            found[key] = {
                "value": int(match.group(1).replace(",", "")),
                "parsed": True,
                "matched_text": match.group(0).strip()[:120],
            }
        else:
            found[key] = {"value": None, "parsed": False}
    return found


def _count_jsonl_lines(path: Path) -> Optional[int]:
    """Count non-empty lines in a JSONL file, or ``None`` if it is absent."""
    if not path.exists():
        return None
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def run_step(step: SoupStep, *, cwd: Path) -> Dict[str, Any]:
    """Execute one Soup step, capturing stdout/stderr verbatim to a log file.

    A non-zero exit is RECORDED, not raised: a failing Soup command is itself an
    audit finding, and the remaining steps still carry information.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.name}.log"

    started = time.time()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            list(step.argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except FileNotFoundError as exc:
        returncode, stdout, stderr = 127, "", f"{exc}"
    duration = round(time.time() - started, 3)

    log_path.write_text(
        f"$ {step.as_str()}\n"
        f"# cwd={cwd}\n"
        f"# exit={returncode} duration={duration}s\n"
        f"# why: {step.why}\n"
        f"{'-' * 72}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    result: Dict[str, Any] = {
        "name": step.name,
        "command": step.as_str(),
        "why": step.why,
        "exit_code": returncode,
        "duration_seconds": duration,
        "log": log_path.relative_to(ROOT).as_posix(),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "parsed": _extract_numbers(stdout + "\n" + stderr),
    }
    if step.writes is not None:
        result["wrote"] = step.writes.relative_to(ROOT).as_posix()
        result["wrote_exists"] = step.writes.exists()
        # Authoritative row count: count the file Soup actually produced rather
        # than trusting a parsed log line.
        result["wrote_rows"] = _count_jsonl_lines(step.writes)
    return result


# --------------------------------------------------------------------------- #
# Help probe — the ONLY sanctioned way a new flag enters the plan
# --------------------------------------------------------------------------- #


def probe_help(soup: str = "soup") -> int:
    """Capture ``--help`` for the candidate subcommands and save it as evidence.

    Rationale: ``soup data`` exposes 44 subcommands in 0.73.3, and this
    repository has only ever verified five of them. Rather than guess at the
    others' flags (which would be fabrication), this dumps their real help text
    to ``data/processed/msa/soup/help/`` so the plan can be extended from
    observed fact.

    Returns:
        A process exit code (0 if every probe succeeded).
    """
    help_dir = OUT_DIR / "help"
    help_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(soup) is None:
        print(
            f"error: '{soup}' is not on PATH. Probe inside the container:\n"
            f"       docker compose run --rm soup-cpu "
            f"python scripts/soup_analyze_msa.py --probe-help",
            file=sys.stderr,
        )
        return 3

    failures = 0
    index: Dict[str, Any] = {}
    targets = [("_data", [soup, "data", "--help"])] + [
        (command, [soup, "data", command, "--help"])
        for command in HELP_PROBE_COMMANDS
    ]

    for name, argv in targets:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        text = completed.stdout + completed.stderr
        (help_dir / f"{name}.txt").write_text(
            f"$ {' '.join(argv)}\n# exit={completed.returncode}\n"
            f"{'-' * 72}\n{text}",
            encoding="utf-8",
        )
        index[name] = {
            "command": " ".join(argv),
            "exit_code": completed.returncode,
            "help_file": (help_dir / f"{name}.txt").relative_to(ROOT).as_posix(),
        }
        if completed.returncode != 0:
            failures += 1
        print(f"  {'ok ' if completed.returncode == 0 else 'ERR'} {name}")
        print(text.rstrip())
        print("-" * 72)

    (help_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nhelp text saved under {help_dir.relative_to(ROOT).as_posix()}/")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _eval_contamination_blocker() -> Dict[str, Any]:

    """Explain why no eval-contamination check is run.

    ``data/manifests/msa.yaml`` lists a contamination check as a required TODO.
    It cannot be satisfied yet, and the honest thing is to say so loudly rather
    than run a check against an empty set and report "no contamination".
    """
    manifest = ROOT / "data" / "manifests" / "eval.yaml"
    detail = (
        "data/manifests/eval.yaml declares all three tracks (tounsibench, "
        "arabizi, retention) with 'sources: []' and 'version: 0' -- it is a "
        "documented skeleton. There is no eval record to compare against, so a "
        "decontamination run would report zero overlap for a trivial reason "
        "(an empty reference set) and would be actively misleading evidence."
    )
    if not manifest.exists():
        detail = "data/manifests/eval.yaml does not exist."
    return {
        "status": "blocked",
        "required_by": "data/manifests/msa.yaml -> TODO "
                       "'Verify record-level disjointness from eval.yaml'",
        "reason": detail,
        "unblocks_when": "eval.yaml declares at least one real source per track",
    }


def write_report(
    preflight: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    *,
    soup_available: bool,
    dry_run: bool,
) -> Path:
    """Write the machine-readable run report and return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "soup_report.json"
    report = {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/soup_analyze_msa.py",
        "slice": "msa_formal",
        "soup_image": "ghcr.io/makazhanalpamys/soup:0.73.3",
        "soup_on_path": soup_available,
        "dry_run": dry_run,
        "soup_format": SOUP_FORMAT,
        "dedup_threshold": DEDUP_THRESHOLD,
        "pools_analyzed": [pool.pool_id for pool in POOLS],
        "pools_excluded": {
            "palm": "blocked: gated (403 GatedRepoError) AND CC-BY-NC-ND-4.0 "
                    "(see data/manifests/msa.yaml -> blocked_candidates)",
            "arsyra-complete": "blocked: Hub artifact is a 50-record preview of "
                               "a paid, non-redistributable corpus",
        },
        "preflight": preflight,
        "steps": steps,
        "eval_contamination_check": _eval_contamination_blocker(),
        "caveats": [
            "Values under steps[].parsed are BEST-EFFORT regex extractions from "
            "Soup's stdout and each carries its own 'parsed' flag. The "
            "authoritative record is the raw log referenced by steps[].log.",
            "steps[].wrote_rows is counted from the file Soup actually wrote "
            "and is therefore authoritative, unlike the parsed values.",
            "preflight[].local_measurements are measured by THIS script, not by "
            "Soup, and are labelled as such.",
            "This run writes to data/processed/msa/soup/ and does not modify the "
            "September 4 artifacts cited in docs/data/msa_audit.md.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _docker_hint(service: str = "soup-cpu") -> str:
    """Return the exact container command that runs this script for real."""
    return (
        f"docker compose run --rm {service} "
        f"python scripts/soup_analyze_msa.py"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python scripts/soup_analyze_msa.py",
        description=(
            "Run the Soup 0.73.3 dataset toolchain over the MSA candidate pools "
            "and write a machine-readable report."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="materialise Soup inputs and print the plan, but run no Soup command",
    )
    parser.add_argument(
        "--in-docker",
        action="store_true",
        help="re-exec this script inside the pinned Soup container",
    )
    parser.add_argument(
        "--service",
        default=os.environ.get("SERVICE", "soup-cpu"),
        help="compose service used by --in-docker (default: soup-cpu)",
    )
    parser.add_argument(
        "--soup",
        default="soup",
        help="Soup executable name (default: soup)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="reuse existing *_alpaca.jsonl instead of regenerating them",
    )
    parser.add_argument(
        "--probe-help",
        action="store_true",
        help=(
            "capture the real --help of the candidate `soup data` subcommands "
            "to data/processed/msa/soup/help/ and exit; this is how a new flag "
            "is allowed to enter the plan (observed, never guessed)"
        ),
    )
    args = parser.parse_args(argv)

    if args.probe_help and not args.in_docker:
        return probe_help(soup=args.soup)

    if args.in_docker:

        command = [
            "docker", "compose", "run", "--rm", args.service,
            "python", "scripts/soup_analyze_msa.py",
        ]
        print(f"==> re-executing inside the Soup container:\n    {' '.join(command)}")
        try:
            return subprocess.run(command, cwd=str(ROOT), check=False).returncode  # noqa: S603
        except FileNotFoundError:
            print("error: 'docker' not found on PATH.", file=sys.stderr)
            return 2

    print("=" * 72)
    print(" Soup analysis — msa_formal candidate pools")
    print("=" * 72)

    # --- 1. preflight ------------------------------------------------------- #
    preflight: List[Dict[str, Any]] = []
    for pool in POOLS:
        if args.skip_prepare and pool.soup_input.exists():
            print(f"[prepare] {pool.pool_id}: reusing {pool.soup_input.name}")
            preflight.append(
                {
                    "pool_id": pool.pool_id,
                    "reused_existing": True,
                    "soup_input": pool.soup_input.relative_to(ROOT).as_posix(),
                    "soup_input_sha256": _sha256(pool.soup_input),
                    "rows_written": _count_jsonl_lines(pool.soup_input),
                }
            )
            continue
        try:
            summary = prepare_pool(pool)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        preflight.append(summary)
        local = summary["local_measurements"]
        print(
            f"[prepare] {pool.pool_id}: {summary['rows_written']} rows -> "
            f"{pool.soup_input.name}  (sha256 {summary['soup_input_sha256'][:12]}…)"
        )
        print(
            f"           local: empty_io={local['empty_instruction_or_output']}  "
            f"unique_passages={local['unique_passages']}  "
            f"max_q_per_passage={local['max_questions_per_passage']}"
        )

    # --- 2. plan ------------------------------------------------------------ #
    plan = build_plan(soup=args.soup)
    plan_path = OUT_DIR / "soup_run_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "soup_format": SOUP_FORMAT,
                "dedup_threshold": DEDUP_THRESHOLD,
                "note": "Only `soup data` flags verified against 0.73.3 in this "
                        "repository are used; see the module docstring.",
                "steps": [
                    {"name": s.name, "command": s.as_str(), "why": s.why}
                    for s in plan
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n--- plan: {len(plan)} Soup invocation(s) "
          f"(written to {plan_path.relative_to(ROOT).as_posix()}) ---")
    for step in plan:
        print(f"  {step.name:34s} {step.as_str()}")

    soup_available = shutil.which(args.soup) is not None

    # --- 3. run ------------------------------------------------------------- #
    if args.dry_run:
        print("\n[dry-run] no Soup command was executed.")
        report_path = write_report(
            preflight, [], soup_available=soup_available, dry_run=True
        )
        print(f"[dry-run] report: {report_path.relative_to(ROOT).as_posix()}")
        return 0

    if not soup_available:
        print(
            f"\nerror: '{args.soup}' is not on PATH, so no Soup number can be "
            f"produced here.\n"
            f"       Soup lives only in the pinned image "
            f"(ghcr.io/makazhanalpamys/soup:0.73.3). Run:\n"
            f"           {_docker_hint(args.service)}\n"
            f"       or re-run this script with --in-docker.\n"
            f"       To materialise inputs and inspect the plan offline, use "
            f"--dry-run.",
            file=sys.stderr,
        )
        write_report(preflight, [], soup_available=False, dry_run=False)
        return 3

    print("\n--- executing ---")
    results: List[Dict[str, Any]] = []
    failures = 0
    for step in plan:
        print(f"  -> {step.name} …", end="", flush=True)
        result = run_step(step, cwd=ROOT)
        results.append(result)
        status = "ok" if result["exit_code"] == 0 else f"EXIT {result['exit_code']}"
        extra = ""
        if result.get("wrote_rows") is not None:
            extra = f"  wrote {result['wrote_rows']} rows"
        print(f" {status} ({result['duration_seconds']}s){extra}")
        if result["exit_code"] != 0:
            failures += 1

    report_path = write_report(
        results and preflight or preflight,
        results,
        soup_available=True,
        dry_run=False,
    )

    print("\n" + "=" * 72)
    print(f" report : {report_path.relative_to(ROOT).as_posix()}")
    print(f" logs   : {LOG_DIR.relative_to(ROOT).as_posix()}/")
    print(f" steps  : {len(results)} run, {failures} non-zero exit(s)")
    print("=" * 72)
    if failures:
        print(
            "NOTE: a non-zero exit is recorded, not hidden. Read the logs above "
            "before quoting any number from this run.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
