"""Print license, revision SHA, size and files for each candidate HF dataset."""
from huggingface_hub import HfApi

CANDIDATES = [
    "Farah21/Tunisia_Tech",
    "Farah21/reddit-tunisia-qa",
    "tunis-ai/MADAR-TUN",
    "chaymafourati/tunizi",
    "hamzabouajila/tunisian-derja-unified-raw-corpus",
    "linagora/Tunisian_Derja_Dataset",
    "linagora/linto-dataset-text-ar-tn",
    "khaled123/Tunisian_Dialectic_English_Derja",
]

api = HfApi()
for repo in CANDIDATES:
    print("=" * 70)
    print(repo)
    try:
        info = api.dataset_info(repo, files_metadata=True)
    except Exception as e:
        print("  ERROR:", type(e).__name__, str(e)[:120])
        continue
    card = info.card_data.to_dict() if info.card_data else {}
    print("  license  :", card.get("license"))
    print("  sha      :", info.sha)
    print("  modified :", info.last_modified)
    print("  tags     :", [t for t in info.tags if not t.startswith("region:")][:12])
    total = sum((s.size or 0) for s in (info.siblings or []))
    print("  files    :", len(info.siblings or []), f"({total/1e6:.1f} MB)")
    for s in (info.siblings or [])[:8]:
        print("    -", s.rfilename, f"{(s.size or 0)/1e6:.1f} MB")