"""
Example 6: Failures — the part that bites in production.

An exception inside a thread does NOT crash your program and does NOT print
anything. It is captured and stored inside the Future. It only surfaces when
someone calls .result(). If nobody calls .result(), the failure is silent.

That is the #1 threading bug in agent code: a tool errored, the future was
never redeemed, and the agent answered confidently from two of three sources.

Three things to get right:
  1. ALWAYS redeem every future you create.
  2. One tool failing must not kill the other four -> catch per future.
  3. A hung call must not hang the agent -> result(timeout=...).

Note the asymmetry in cancellation: result(timeout=) stops YOU waiting.
It does not stop the thread. Python cannot kill a running thread — the only
real timeout is the one you pass to the HTTP client itself.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError


def good_tool(name):
    time.sleep(0.5)
    return f"{name}: ok"


def broken_tool(name):
    time.sleep(0.3)
    raise ValueError(f"{name}: upstream returned 503")


def hanging_tool(name):
    time.sleep(30)                         # a service that never answers
    return f"{name}: finally"


if __name__ == "__main__":
    # --- The silent failure ---
    print("=== Failure with no .result() call ===")
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(broken_tool, "inventory")     # receipt thrown away
        time.sleep(0.6)
    print("  Program still running. No traceback. The error is simply gone.")
    print("  Nothing above this line told you inventory lookup failed.\n")

    # --- Redeeming the future surfaces it ---
    print("=== Same call, but we redeem the future ===")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f = ex.submit(broken_tool, "inventory")
        try:
            f.result()
        except ValueError as e:
            print(f"  Caught in the MAIN thread: {e}")
    print()

    # --- Isolating failures across a fan-out ---
    print("=== Fan-out where one tool is broken ===")
    jobs = {"orders": good_tool, "inventory": broken_tool,
            "policy": good_tool, "shipping": good_tool}

    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn, name): name for name, fn in jobs.items()}
        for f in as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()      # per-future try/except
            except Exception as e:
                errors[name] = str(e)           # degrade, don't collapse

    print(f"  succeeded: {sorted(results)}")
    print(f"  failed:    {errors}")
    print("  The agent can still answer from 3 of 4 sources — and it KNOWS")
    print("  which one is missing, so it can say so instead of guessing.\n")

    # --- Timeouts ---
    print("=== A tool that hangs ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_fast = ex.submit(good_tool, "policy")
        f_slow = ex.submit(hanging_tool, "legacy-crm")

        print(f"  policy -> {f_fast.result()}")
        try:
            f_slow.result(timeout=2)
        except TimeoutError:
            print(f"  legacy-crm -> gave up after {time.time() - start:.1f}s")

        f_slow.cancel()                    # only works if it has NOT started yet
        print(f"  cancel() succeeded? {f_slow.cancelled()}  (it was already running)")
        print("  ...and the `with` block now waits for that 30s thread anyway.")

    print(f"  Block exited at {time.time() - start:.1f}s")
    print()
    print("Lesson: the real fix is a timeout on the request itself —")
    print("  requests.get(url, timeout=5)   or   OpenAI(timeout=20)")
    print("so the thread actually ends. result(timeout=) only frees the caller.")
