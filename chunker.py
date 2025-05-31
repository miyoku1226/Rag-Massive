from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def chunk(messages: List[Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    for m in messages:
        parts = _splitter.split_text(m["body"])
        for idx, part in enumerate(parts):
            chunks.append({
                "id": f"{m['id']}_{idx}",
                "text": part,
                "metadata": {
                    "orig_id": m["id"],
                    "author": m["author"],
                    "timestamp": m["timestamp"],
                    "conversation_id": m["conversation_id"],
                },
            })
    return chunks