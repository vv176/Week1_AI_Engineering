"""
Example 10: The shape you will meet in the real codebase — DIVIDE and CONQUER.

This is the skeleton of the retrieval pipeline we ship in the support-copilot
project (sessions/s9_rag_final/pipeline.py):

    DIVIDE   run three retrieval strategies in parallel. They search in
             different ways — keyword, semantic, relational — so they find
             overlapping but not identical sets of chunks.

    CONQUER  once all three have returned, merge them into ONE pool keyed by
             doc_id, remembering which strategies found each document.

Note where the merge happens: in the MAIN thread, after every worker has
returned. No locks, no shared dict, no race — the rule from Example 7.
Workers COMPUTE and RETURN; the main thread MERGES.

The payoff of the merge is not just deduplication. A document that comes back
from two independent retrievers is a stronger match than one that comes back
from a single retriever, and you only learn that by combining them.
"""

import time
from concurrent.futures import ThreadPoolExecutor

# Each strategy waits on a different backend, so each has its own latency.
LATENCY = {"hybrid": 1.2, "hyde": 2.0, "graph": 1.5}


# ---------- the three strategies: each returns a list of (doc_id, text) ----------

def hybrid_strategy(query):
    """Keyword + vector search over the chunk index."""
    time.sleep(LATENCY["hybrid"])
    return [
        ("doc_refund_policy", "Refunds are accepted within 30 days."),
        ("doc_shipping_faq", "Orders ship within 2 business days of payment."),
    ]


def hyde_strategy(query):
    """Ask an LLM to write a hypothetical answer, then search with THAT."""
    time.sleep(LATENCY["hyde"])
    return [
        ("doc_refund_policy", "Refunds are accepted within 30 days of delivery, "
                              "provided the item is unused and in original packaging."),
        ("doc_returns_howto", "To start a return, open the app and tap Orders > Return."),
    ]


def graph_strategy(query):
    """Walk the entity graph — finds docs that don't read like the query at all."""
    time.sleep(LATENCY["graph"])
    return [
        ("doc_furniture_exception", "Furniture has a 15-day return window and a "
                                    "pickup fee of Rs.500."),
    ]


STRATEGIES = [
    ("hybrid", hybrid_strategy),
    ("hyde", hyde_strategy),
    ("graph", graph_strategy),
]


def unified_search(query):
    # --- DIVIDE: all three run at once ---
    print(f"DIVIDE: running {len(STRATEGIES)} strategies in parallel")
    start = time.time()

    with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as ex:
        futures = {name: ex.submit(fn, query) for name, fn in STRATEGIES}
        results = {name: f.result() for name, f in futures.items()}

    elapsed = time.time() - start
    for name, chunks in results.items():
        print(f"    {name:8} returned {len(chunks)} chunk(s): "
              f"{[doc_id for doc_id, _ in chunks]}")
    print(f"  took {elapsed:.1f}s — sequentially it would be "
          f"{sum(LATENCY.values()):.1f}s\n")

    # --- CONQUER: union by doc_id, in the main thread, after everyone returned ---
    print("CONQUER: merging by doc_id")
    merged = {}
    for name, chunks in results.items():
        for doc_id, text in chunks:
            if doc_id not in merged:
                merged[doc_id] = {"text": text, "sources": []}
            merged[doc_id]["sources"].append(name)
            if len(text) > len(merged[doc_id]["text"]):
                merged[doc_id]["text"] = text          # keep the fuller snippet

    total_chunks = sum(len(c) for c in results.values())
    print(f"  {total_chunks} chunks in -> {len(merged)} unique docs out\n")

    # Docs found by the most strategies first — agreement is a relevance signal.
    return sorted(merged.items(), key=lambda kv: len(kv[1]["sources"]), reverse=True)


if __name__ == "__main__":
    query = "can I return a desk I bought two weeks ago?"
    print(f"Query: {query!r}\n")

    docs = unified_search(query)

    print("Merged pool:")
    for doc_id, doc in docs:
        found_by = ", ".join(doc["sources"])
        print(f"\n  {doc_id}   (found by {len(doc['sources'])}: {found_by})")
        print(f"    {doc['text']}")

    print()
    print("Read the output top-down:")
    print("  doc_refund_policy came back from TWO strategies — that agreement is")
    print("  itself evidence, and we kept hyde's longer snippet over hybrid's.")
    print("  doc_furniture_exception came ONLY from the graph strategy, and it is")
    print("  the one document that actually answers the question. Drop a strategy")
    print("  to save time and you lose it entirely.")
    print()
    print("That is the argument for threads here: running all three costs")
    print(f"  {max(LATENCY.values()):.1f}s instead of {sum(LATENCY.values()):.1f}s, "
          "so you never have to choose between them.")
    print()
    print("Next: open sessions/s9_rag_final/pipeline.py and read unified_search().")
    print("Same skeleton, plus an LLM reranker and an abstention gate.")
