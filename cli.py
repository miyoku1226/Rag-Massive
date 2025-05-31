import argparse, json, sys, os, tempfile, shutil
from pathlib import Path
from data_loader import load_missive
from chunker import chunk 
from embed_store import build_index
from rag import answer

def main():
    parser = argparse.ArgumentParser(description="Missive RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("zip", help="Missive export .zip file")
    ing.add_argument("--out", default="messages.jsonl", help="output .jsonl file")

    ask = sub.add_parser("ask")
    ask.add_argument("question", help="Your question")
    ask.add_argument("--db", default=".chroma", help="Path to Chroma persistence directory")

    args = parser.parse_args()

    if args.cmd == "ingest":
        msgs = load_missive(args.zip)
        with open(args.out, "w", encoding="utf-8") as fw:
            for m in msgs:
                fw.write(json.dumps(m, ensure_ascii=False) + "\n")
        chunks = chunk(msgs)
        build_index(chunks)
        print(f"Parsed {len(msgs)} messages and built the vector index.")

    elif args.cmd == "ask":
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        db = Chroma(persist_directory=args.db, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
        print(answer(db, args.question))

if __name__ == "__main__":
    main()
