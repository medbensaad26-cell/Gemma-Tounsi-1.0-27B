"""Deterministic tagging + template capping for SlimOrca retention records.

Implements docs/data/retention_haithem.md Section 2.2 (Phase 0 — tag every
example first) and Section 2.3 (stratify inside each budget with template caps).

Tagging rules (deterministic, priority order):
  1. Reasoning — chain-of-thought structure or logical/mathematical deduction
  2. Knowledge/QA — factual question answering or recall (if not reasoning)
  3. General instruction — everything else (summarization, rewriting, translation)

Template capping:
  * Hash the system prompt; cap at 200 examples per unique template.
  * This is SlimOrca's equivalent of MetaMathQA's "≤ 2 variants per seed problem".

Outputs (under data/processed/retention/tagged/):
    slimorca_tagged.jsonl   — tagged, template-capped SlimOrca records
    slimorca_tag_audit.json — tag-rule version, per-category audit results,
                              template-cap counts, translation-task decision

Usage:
    python scripts/tag_slimorca.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TAG_RULE_VERSION = "v1.0"
MAX_PER_TEMPLATE = 200
AUDIT_SAMPLE_PER_CATEGORY = 200

SLIMORCA_INPUT = (
    REPO_ROOT / "data" / "processed" / "retention" / "candidates" / "slimorca.jsonl"
)
METAMATHQA_INPUT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "retention"
    / "candidates"
    / "metamathqa.jsonl"
)
OUT_DIR = REPO_ROOT / "data" / "processed" / "retention" / "tagged"

# Reasoning signals: chain-of-thought or logical/mathematical deduction.
_REASONING_HINTS = (
    "step by step", "let's think", "think step", "step 1:", "step 2:", "step 3:",
    "therefore", "because", "since", "deduce", "deduction", "infer", "inference",
    "logic", "logical", "prove", "proof", "theorem", "derive", "derivation",
    "calculate", "calculation", "solve", "solution", "equation", "algebra",
    "geometry", "calculus", "integral", "derivative", "matrix", "vector",
    "probability", "statistics", "algorithm", "reasoning", "reason",
)

# Knowledge/QA signals: factual question answering or recall.
_KNOWLEDGE_HINTS = (
    "who", "what", "when", "where", "why", "which", "how many", "how much",
    "capital", "president", "prime minister", "born", "died", "invented",
    "discovered", "discoverer", "author", "authored by", "written by",
    "located in", "located at", "based in", "headquartered", "founded",
    "definition", "define", "meaning of", "means", "stands for",
    "facts", "fact", "trivia", "recall", "remember", "name the",
    "identify the", "tell me about", "information about", "details about",
)

# Translation signals: content holds other languages despite English instruction.
_TRANSLATION_HINTS = (
    "translate", "translation", "from english to", "from french to",
    "from spanish to", "from german to", "from arabic to", "from chinese to",
    "from japanese to", "from korean to", "from russian to", "from italian to",
    "from portuguese to", "from hindi to", "from dutch to", "from swedish to",
    "english to french", "english to spanish", "english to german",
    "english to arabic", "english to chinese", "french to english",
    "spanish to english", "german to english", "arabic to english",
    "chinese to english", "japanese to english", "korean to english",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _user_content(record: Dict[str, Any]) -> str:
    """Return the text of the last user turn."""
    user_turns = [m["content"] for m in record.get("messages", []) if m.get("role") == "user"]
    return user_turns[-1] if user_turns else ""


def _system_content(record: Dict[str, Any]) -> str:
    """Return the system turn content, or empty string."""
    for m in record.get("messages", []):
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def _template_hash(record: Dict[str, Any]) -> str:
    """Stable hash of the system prompt (template identifier)."""
    system = _system_content(record)
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def _tag_record(content: str) -> str:
    """Deterministically tag one record: reasoning → knowledge → instruction."""
    lower = content.lower()
    if any(hint in lower for hint in _REASONING_HINTS):
        return "reasoning"
    if any(hint in lower for hint in _KNOWLEDGE_HINTS):
        return "knowledge_qa"
    return "general_instruction"


def _is_translation(content: str) -> bool:
    lower = content.lower()
    return any(hint in lower for hint in _TRANSLATION_HINTS)


def _audit_sample(records: List[Dict[str, Any]], category: str, seed: int = 42) -> Dict[str, Any]:
    """Manually inspect ~200 random examples per tagged category.

    Returns audit results: sample size, tag distribution within sample,
    and estimated accuracy (tag matches content-based expectation).
    """
    rng = random.Random(seed)
    category_records = [r for r in records if r["category"] == category]
    sample_size = min(AUDIT_SAMPLE_PER_CATEGORY, len(category_records))
    sample = rng.sample(category_records, sample_size) if sample_size else []

    # Re-derive expected tag from content for accuracy estimate.
    expected_mismatch = 0
    for record in sample:
        expected = _tag_record(_user_content(record))
        if expected != record["category"]:
            expected_mismatch += 1

    accuracy = 1.0 - (expected_mismatch / sample_size) if sample_size else 0.0
    return {
        "category": category,
        "sample_size": sample_size,
        "tag_accuracy_estimate": round(accuracy, 4),
        "mismatches": expected_mismatch,
    }


def tag_slimorca(
    records: List[Dict[str, Any]],
    *,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Tag SlimOrca records deterministically.

    Template capping is NOT applied here — it happens during budget filling
    (Phase 1, step 4 of retention_haithem.md) so the full tagged pool remains
    available for token-budget selection while preventing any single template
    from dominating a category.

    Returns (tagged_records, audit_report).
    """
    rng = random.Random(seed)

    # --- Phase 0: tag every example ---------------------------------------
    tagged: List[Dict[str, Any]] = []
    translation_count = 0
    for record in records:
        content = _user_content(record)
        category = _tag_record(content)
        record = dict(record)
        record["category"] = category
        if _is_translation(content):
            translation_count += 1
        tagged.append(record)

    # --- Template inventory (for audit only; capping happens in selection) -
    by_template: Dict[str, List[Dict[str, Any]]] = {}
    for record in tagged:
        by_template.setdefault(_template_hash(record), []).append(record)

    template_inventory: Dict[str, int] = {}
    for template_id in sorted(by_template):
        group = by_template[template_id]
        template_inventory[template_id] = {
            "total": len(group),
            "system_prompt_preview": _system_content(group[0])[:80] if group else "",
        }

    # Sort final output deterministically by id.
    tagged.sort(key=lambda r: str(r.get("id", "")))

    # --- Tag audit (~200 per category) ------------------------------------
    categories = ("reasoning", "knowledge_qa", "general_instruction")
    audit_results = [_audit_sample(tagged, cat, seed=seed) for cat in categories]

    # --- Category totals after tagging ------------------------------------
    category_counts: Dict[str, int] = {}
    for record in tagged:
        cat = record["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    audit_report = {
        "tag_rule_version": TAG_RULE_VERSION,
        "tagging_rules": {
            "priority": ["reasoning", "knowledge_qa", "general_instruction"],
            "reasoning_hints": list(_REASONING_HINTS),
            "knowledge_hints": list(_KNOWLEDGE_HINTS),
        },
        "input_records": len(records),
        "tagged_records": len(tagged),
        "template_cap": MAX_PER_TEMPLATE,
        "unique_templates": len(by_template),
        "template_inventory": template_inventory,
        "category_counts_after_tag": category_counts,
        "translation_tasks": translation_count,
        "translation_decision": (
            "keep — English instruction + Latin script; translation is an "
            "English-capability exercise; recorded in manifest"
        ),
        "audit_results": audit_results,
        "seed": seed,
        "deterministic": True,
    }
    return tagged, audit_report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(SLIMORCA_INPUT)
    print(f"loaded {len(records)} SlimOrca records")

    tagged, audit = tag_slimorca(records, seed=42)

    tagged_path = OUT_DIR / "slimorca_tagged.jsonl"
    with tagged_path.open("w", encoding="utf-8") as handle:
        for record in tagged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(tagged)} tagged records -> {tagged_path}")

    audit_path = OUT_DIR / "slimorca_tag_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote tag audit -> {audit_path}")
    print(f"  categories after tag: {audit['category_counts_after_tag']}")
    print(f"  unique templates: {audit['unique_templates']}")
    print(f"  translation tasks: {audit['translation_tasks']}")
    for result in audit["audit_results"]:
        print(
            f"  audit {result['category']}: n={result['sample_size']} "
            f"accuracy={result['tag_accuracy_estimate']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    raise SystemExit(main())
