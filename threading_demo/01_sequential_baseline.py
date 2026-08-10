"""
Example 1: The problem — an agent doing three lookups, one after another.

An agent almost never needs just one thing. To answer "where is my refund?"
it needs the order record, the shipping status, and the refund policy.
Three independent lookups. None of them depends on the other two.

Done sequentially, the user waits for the SUM of all three.

Notice what is actually slow here: nothing is computing. Every second is spent
WAITING — on a network, a database, another machine. Hold on to that fact.
It is the entire justification for threads.
"""

import time


def fetch_order(order_id):
    print(f"  -> fetch_order({order_id}) started")
    time.sleep(2)                          # network wait, not CPU work
    print(f"  <- fetch_order({order_id}) done")
    return {"order_id": order_id, "total": 4999}


def fetch_shipping(order_id):
    print(f"  -> fetch_shipping({order_id}) started")
    time.sleep(3)                          # a slower service
    print(f"  <- fetch_shipping({order_id}) done")
    return {"status": "in_transit", "eta_days": 2}


def fetch_refund_policy(order_id):
    print(f"  -> fetch_refund_policy({order_id}) started")
    time.sleep(1)
    print(f"  <- fetch_refund_policy({order_id}) done")
    return {"window_days": 30}


if __name__ == "__main__":
    order_id = "ORD-7781"

    print("=== Sequential: one tool call at a time ===")
    start = time.time()

    order = fetch_order(order_id)          # 2s
    shipping = fetch_shipping(order_id)    # then 3s
    policy = fetch_refund_policy(order_id) # then 1s

    elapsed = time.time() - start

    print()
    print(f"order:    {order}")
    print(f"shipping: {shipping}")
    print(f"policy:   {policy}")
    print()
    print(f"Total time: {elapsed:.1f}s   (2 + 3 + 1 = 6)")
    print()
    print("The slowest call was 3s. The user waited 6s.")
    print("The extra 3s bought us nothing — we were just standing in a queue.")
