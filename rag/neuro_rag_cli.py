#!/usr/bin/env python3
"""
neuro_rag_cli.py — Manage Neuro's RAG system from the terminal
================================================================
Usage:
  python neuro_rag_cli.py stats
  python neuro_rag_cli.py sync
  python neuro_rag_cli.py list
  python neuro_rag_cli.py add <file_or_dir>
  python neuro_rag_cli.py delete <filename>
  python neuro_rag_cli.py search <query>
  python neuro_rag_cli.py clear-memory
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)   # keep output clean

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.rag_engine import NeuroRAG, KNOWLEDGE_DIR


def cmd_stats(rag: NeuroRAG, _args):
    s = rag.stats()
    print("\n── Neuro RAG Stats ──────────────────")
    print(f"  Conversation memories : {s['memory_count']}")
    print(f"  Knowledge chunks      : {s['knowledge_count']}")
    print(f"  Knowledge folder      : {s['knowledge_dir']}")
    print(f"  ChromaDB path         : {s['chroma_dir']}")
    print()


def cmd_sync(rag: NeuroRAG, _args):
    print("Scanning knowledge/ folder …")
    n = rag.sync_knowledge_folder()
    print(f"Done — {n} new chunks indexed.")


def cmd_list(rag: NeuroRAG, _args):
    results = rag._knowledge.get(include=["metadatas"])
    if not results["ids"]:
        print("Knowledge base is empty.")
        return

    sources = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\n── Knowledge Base Files ─────────────")
    for src, count in sorted(sources.items()):
        print(f"  {src:40s}  {count} chunks")
    print(f"\nTotal: {sum(sources.values())} chunks across {len(sources)} files\n")


def cmd_add(rag: NeuroRAG, args):
    target = Path(args.path)
    if not target.exists():
        print(f"Error: '{target}' does not exist.")
        sys.exit(1)

    if target.is_dir():
        for f in target.rglob("*"):
            if f.is_file():
                _add_file(rag, f)
    else:
        _add_file(rag, target)


def _add_file(rag: NeuroRAG, src: Path):
    import shutil
    dest = KNOWLEDGE_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        print(f"Copied '{src.name}' → knowledge/")
    n = rag.index_file(dest)
    if n:
        print(f"Indexed '{src.name}' → {n} chunks")
    else:
        print(f"'{src.name}' already up-to-date, skipped.")


def cmd_delete(rag: NeuroRAG, args):
    rag.delete_knowledge(args.filename)
    target = KNOWLEDGE_DIR / args.filename
    if target.exists():
        target.unlink()
        print(f"Deleted file '{args.filename}' from knowledge/")
    print(f"Removed all chunks for '{args.filename}' from the vector DB.")


def cmd_search(rag: NeuroRAG, args):
    query = " ".join(args.query)
    print(f"\nSearching for: '{query}'\n")

    mem_hits  = rag.retrieve_memory(query, top_k=3)
    know_hits = rag.retrieve_knowledge(query, top_k=3)

    if mem_hits:
        print("── Memory hits ──────────────────────")
        for h in mem_hits:
            score = round(1 - h["distance"], 3)
            ts    = h["metadata"].get("timestamp", "")
            print(f"  [{score}] {ts}\n  {h['document'][:200]}…\n")
    else:
        print("── Memory hits ──────────────────────\n  (none)\n")

    if know_hits:
        print("── Knowledge hits ───────────────────")
        for h in know_hits:
            score = round(1 - h["distance"], 3)
            src   = h["metadata"].get("source", "?")
            print(f"  [{score}] {src}\n  {h['document'][:200]}…\n")
    else:
        print("── Knowledge hits ───────────────────\n  (none)\n")


def cmd_clear_memory(rag: NeuroRAG, _args):
    confirm = input("This will delete ALL conversation memories. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        ids = rag._memory.get()["ids"]
        if ids:
            rag._memory.delete(ids=ids)
        print(f"Cleared {len(ids)} memory entries.")
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        description="Neuro RAG management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stats",        help="Show RAG statistics")
    sub.add_parser("sync",         help="Sync knowledge/ folder")
    sub.add_parser("list",         help="List indexed knowledge files")
    sub.add_parser("clear-memory", help="Clear all conversation memories")

    p_add = sub.add_parser("add",  help="Add a file or folder to the knowledge base")
    p_add.add_argument("path")

    p_del = sub.add_parser("delete", help="Remove a file from the knowledge base")
    p_del.add_argument("filename")

    p_search = sub.add_parser("search", help="Search the RAG system")
    p_search.add_argument("query", nargs="+")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    rag = NeuroRAG()

    dispatch = {
        "stats":        cmd_stats,
        "sync":         cmd_sync,
        "list":         cmd_list,
        "add":          cmd_add,
        "delete":       cmd_delete,
        "search":       cmd_search,
        "clear-memory": cmd_clear_memory,
    }
    dispatch[args.cmd](rag, args)


if __name__ == "__main__":
    main()
