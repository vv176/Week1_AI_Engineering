"""
Example 3: ThreadPoolExecutor — the version you will actually write.

Two problems with raw threads:
  1. Functions can't return values, so you pass a shared dict around.
  2. Spawning one thread per item breaks at 5,000 items.

A pool fixes both. You hand it work; it keeps a fixed number of workers busy
and hands you back a Future — a receipt you can redeem for the return value.

  ex.submit(fn, arg)  -> Future        (fire it off, keep the receipt)
  future.result()     -> return value  (blocks until that one is ready)
  ex.map(fn, items)   -> results in input order (the convenience form)

`with ThreadPoolExecutor(...) as ex:` waits for every submitted task at the
end of the block. You almost never call join() yourself again.
"""

import time
from concurrent.futures import ThreadPoolExecutor


def fetch_order(order_id):
    time.sleep(2)
    return {"order_id": order_id, "total": 4999}


def fetch_shipping(order_id):
    time.sleep(3)
    return {"status": "in_transit", "eta_days": 2}


def fetch_refund_policy(order_id):
    time.sleep(1)
    return {"window_days": 30}


def fetch_price(sku):
    time.sleep(1)
    return f"{sku}: Rs.{len(sku) * 100}"


if __name__ == "__main__":
    order_id = "ORD-7781"

    # --- Form A: submit + result. Different functions, mixed arguments. ---
    print("=== submit(): fan out three different tools ===")
    start = time.time()

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_order = ex.submit(fetch_order, order_id)          # started, not awaited
        f_ship = ex.submit(fetch_shipping, order_id)
        f_policy = ex.submit(fetch_refund_policy, order_id)

        print(f"  submit() returned in {time.time() - start:.3f}s -> {f_order}")

        order = f_order.result()                            # blocks until ready
        shipping = f_ship.result()
        policy = f_policy.result()

    print(f"  order:    {order}")
    print(f"  shipping: {shipping}")
    print(f"  policy:   {policy}")
    print(f"  Total: {time.time() - start:.1f}s")

    print()

    # --- Form B: map. Same function, many inputs. Results in INPUT order. ---
    print("=== map(): same tool over many inputs ===")
    skus = ["TSHIRT-M", "MUG", "LAPTOP-STAND", "CABLE-2M", "NOTEBOOK"]
    start = time.time()

    with ThreadPoolExecutor(max_workers=5) as ex:
        prices = list(ex.map(fetch_price, skus))            # 5 x 1s, run together

    for p in prices:
        print(f"  {p}")
    print(f"  Total: {time.time() - start:.1f}s for {len(skus)} lookups")

    print()

    # --- max_workers is the throttle, not a suggestion ---
    print("=== max_workers=2 with 5 tasks: work QUEUES ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(fetch_price, skus))
    print(f"  Total: {time.time() - start:.1f}s  (5 tasks / 2 workers -> 3 waves)")
    print()
    print("Pick max_workers from what the OTHER side tolerates —")
    print("the API's rate limit, the DB's connection pool. Not from your CPU count.")
