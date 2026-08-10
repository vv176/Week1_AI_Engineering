"""
Example 2: The raw primitive — threading.Thread.

A thread is an independent line of execution inside the same process.
Same memory, same variables, separate "cursor" walking through the code.

Two calls matter:
  t.start()   -> begin running the function in a new thread; returns IMMEDIATELY
  t.join()    -> block here until that thread finishes

This is the primitive. It is also slightly awkward — start() gives you nothing
back, so results have to be written into a shared container by hand.
That awkwardness is exactly what ThreadPoolExecutor removes in Example 3.
"""

import time
import threading


def fetch_order(order_id, results):
    time.sleep(2)
    results["order"] = {"order_id": order_id, "total": 4999}


def fetch_shipping(order_id, results):
    time.sleep(3)
    results["shipping"] = {"status": "in_transit", "eta_days": 2}


def fetch_refund_policy(order_id, results):
    time.sleep(1)
    results["policy"] = {"window_days": 30}


if __name__ == "__main__":
    order_id = "ORD-7781"
    results = {}                           # the shared drop-box for return values

    print("=== Threaded: three lookups at the same time ===")
    start = time.time()

    threads = [
        threading.Thread(target=fetch_order, args=(order_id, results)),
        threading.Thread(target=fetch_shipping, args=(order_id, results)),
        threading.Thread(target=fetch_refund_policy, args=(order_id, results)),
    ]

    for t in threads:
        t.start()                          # returns instantly — does NOT wait
    print(f"  All 3 started after {time.time() - start:.3f}s")
    print(f"  Live threads right now: {threading.active_count()}")

    for t in threads:
        t.join()                           # NOW we wait — for all of them together

    elapsed = time.time() - start

    print()
    for key, value in results.items():
        print(f"{key:9}: {value}")
    print()
    print(f"Total time: {elapsed:.1f}s   (max(2, 3, 1) = 3, not 2 + 3 + 1 = 6)")
    print()
    print("Sequential cost = SUM of the waits.")
    print("Parallel cost   = the SLOWEST wait.")
    print()
    print("Note the shape of the problem: start() returns nothing, so we had to")
    print("pass a dict in and have each function write into it. Ugly. Fix: next file.")
