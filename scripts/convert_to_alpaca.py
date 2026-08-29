import json

input_file = "data/raw/meta-math__MetaMathQA/MetaMathQA-395K.jsonl"
output_file = "data/raw/meta-math__MetaMathQA/MetaMathQA-395K-alpaca.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        # Rename to Soup's expected 'alpaca' format keys
        alpaca_row = {
            "instruction": row.get("query", ""),
            "output": row.get("response", ""),
            "type": row.get("type", ""),
            "original_question": row.get("original_question", "")
        }
        fout.write(json.dumps(alpaca_row, ensure_ascii=False) + "\n")

print(f"✅ Converted to alpaca format: {output_file}")