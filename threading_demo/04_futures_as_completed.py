"""
Example 4: Futures and as_completed — react to results as they land.

A Future is a receipt for a value that does not exist yet.
  f.done()      -> is it ready? (never blocks)
  f.result()    -> give me the value (blocks until ready, re-raises exceptions)

If you call result() in submit order, you are pinned to the order you happened
to write, not the order the answers arrive in. as_completed() flips that:
it yields each future the moment IT finishes.

Why an agent cares: this is how you stream partial progress. "Searched the
docs (0.4s)... checked the order DB (0.9s)..." while the slow tool is still
running. Same total time, far better perceived latency — exactly the argument
we made for streaming.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TOOLS = {
    "search_docs":     0.5,
    "get_order":       2.5,
    "check_inventory": 1.0,
    "read_policy":     0.2,
}


def call_tool(name):
    time.sleep(TOOLS[name])                # each tool has its own latency
    return f"result of {name}"


if __name__ == "__main__":
    # --- Ordered: results collected in SUBMIT order ---
    print("=== result() in submit order ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(call_tool, name) for name in TOOLS]
        for name, f in zip(TOOLS, futures):
            value = f.result()                          # blocks in submit order
            print(f"  [{time.time() - start:4.1f}s] {name:16} -> {value}")
    print(f"  Total: {time.time() - start:.1f}s")
    print("  Note: read_policy finished at 0.2s but we only PRINTED it after")
    print("  get_order came back at 2.5s. We blocked on the slow one first.")

    print()

    # --- as_completed: results in FINISH order ---
    print("=== as_completed(): whoever finishes first ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        future_to_name = {ex.submit(call_tool, name): name for name in TOOLS}

        for f in as_completed(future_to_name):          # yields as they land
            name = future_to_name[f]                    # map the receipt back
            value = f.result()                          # already done — no wait
            print(f"  [{time.time() - start:4.1f}s] {name:16} -> {value}")
    print(f"  Total: {time.time() - start:.1f}s")

    print()
    print("Same total time. Completely different feel.")
    print()
    print("The dict {future: name} is the standard trick — a Future does not")
    print("remember what it was made from, so you keep that mapping yourself.")
