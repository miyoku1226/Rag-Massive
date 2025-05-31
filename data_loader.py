import json
import zipfile
from pathlib import Path
from typing import List, Dict

def load_missive(zip_path: str | Path) -> List[Dict]:
    msgs: List[Dict] = []
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                with zf.open(name) as fp:
                    data = json.load(fp)
                    # Missive export could be list or dict; support both
                    if isinstance(data, dict):
                        data = data.get("messages", [])
                    for m in data:
                        # skip if has attachments with images
                        if any(att.get("contentType", "").startswith("image/") for att in m.get("attachments", [])):
                            continue
                        msgs.append({
                            "id": m.get("id"),
                            "body": m.get("body", ""),
                            "timestamp": m.get("createdAt"),
                            "author": m.get("author", {}).get("displayName", "unknown"),
                            "conversation_id": m.get("conversationId"),
                        })
    return msgs

if __name__ == "__main__":
    import argparse, json, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="Path to Missive export zip")
    parser.add_argument("-o", "--out", default="messages.jsonl", help="output .jsonl file")
    args = parser.parse_args()
    msgs = load_missive(args.zip)
    with open(args.out, "w", encoding="utf-8") as fw:
        for m in msgs:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved {len(msgs)} messages to {args.out}")
