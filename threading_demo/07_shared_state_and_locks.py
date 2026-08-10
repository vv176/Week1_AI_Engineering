"""
Example 7: Shared state — where threads stop being free.

Threads share memory. That is why they are cheap, and it is why they are
dangerous. `total += n` is not one step; it is three:

    read total  ->  add n  ->  write total

A thread can be paused between any two of those. Two threads read 100, both
add 50, both write 150. One update vanished. Nobody raises an exception —
you just get a slightly wrong number, sometimes, under load.

Three fixes, in order of preference:
  1. DON'T SHARE. Return values; merge in the main thread once. (best)
  2. Lock the update. Correct, costs a little contention.
  3. Use a thread-safe container (queue.Queue) when you need a live pipeline.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor

TOKENS_PER_CALL = 50
N_CALLS = 200

usage_total = 0                            # shared mutable state — the hazard


def record_unsafe(_):
    global usage_total
    current = usage_total                  # READ
    time.sleep(0)                          # yield to another thread (widens the race)
    usage_total = current + TOKENS_PER_CALL  # WRITE — may overwrite someone's update


lock = threading.Lock()


def record_locked(_):
    global usage_total
    with lock:                             # only one thread inside at a time
        current = usage_total
        time.sleep(0)
        usage_total = current + TOKENS_PER_CALL


def call_llm(i):
    """The preferred shape: touch nothing shared, just return your result."""
    time.sleep(0.01)
    return {"call": i, "tokens": TOKENS_PER_CALL}


if __name__ == "__main__":
    expected = N_CALLS * TOKENS_PER_CALL

    # --- 1. The race ---
    print("=== Unsafe: shared counter, no lock ===")
    usage_total = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(record_unsafe, range(N_CALLS)))
    print(f"  expected {expected}, got {usage_total}   lost {expected - usage_total}")
    print("  Run it again — you get a different wrong number. That is the tell.\n")

    # --- 2. The lock ---
    print("=== Safe: same counter, guarded by a Lock ===")
    usage_total = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(record_locked, range(N_CALLS)))
    print(f"  expected {expected}, got {usage_total}   ({time.time() - start:.2f}s)")
    print("  Correct every time. The lock serialises that one line.\n")

    # --- 3. The pattern you should actually reach for ---
    print("=== Best: don't share at all — return, then merge ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(call_llm, range(N_CALLS)))

    total = sum(r["tokens"] for r in results)     # single-threaded, no race possible
    print(f"  expected {expected}, got {total}   ({time.time() - start:.2f}s)")
    print("  No lock, no global, no bug. The pool already collected everything.\n")

    print("Rule of thumb for agent code:")
    print("  worker threads COMPUTE and RETURN; the main thread MERGES.")
    print("  Reach for a Lock only when a live shared resource forces you to.")
    print()
    print("Footnote: list.append() and dict[k] = v are individually atomic in")
    print("CPython, so collecting into a list from threads is safe. Read-modify-")
    print("write (+=, 'if k not in d: d[k] = ...') is NOT. Don't rely on the")
    print("distinction — it is a CPython detail, not a language guarantee.")
