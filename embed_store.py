import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

def build_index(chunks: List[Dict], persist_directory: str = ".chroma"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]
    db = Chroma(
        collection_name="missive",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    db.persist()
    return db

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="messages.jsonl from data_loader.py")
    args = parser.parse_args()
    chunks = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    db = build_index(chunks)
    print("Indexed", db._collection.count(), "chunks → .chroma/")