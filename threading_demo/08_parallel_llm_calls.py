"""
Example 8: Real OpenAI calls, in parallel.

Everything so far used time.sleep() to fake a wait. Now the wait is real —
a network round trip to OpenAI. Same code, same speedup, because an LLM call
is the most I/O-bound thing in your entire stack.

The pattern here is FAN-OUT: one job, split into N independent sub-questions,
answered concurrently, then stitched together. Map-reduce for LLMs.
You will use this for batch classification, per-document summarisation,
multi-aspect extraction, and self-consistency sampling.

One rule: the OpenAI client is thread-safe. Create it ONCE, share it across
threads. Creating a client per thread means a new connection pool per thread.

Run:  python 08_parallel_llm_calls.py
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)   # one shared client
MODEL = "gpt-4o-mini"

REVIEWS = [
    "Battery dies in 3 hours. Support never replied to my ticket.",
    "Arrived a day early and the build quality is genuinely excellent.",
    "It works. Nothing special, nothing broken.",
    "Third replacement unit and the screen still flickers. Refund please.",
    "Setup took 2 minutes. My whole team switched over.",
    "Good hardware, but the mobile app is unusable on Android.",
]


def classify(review):
    """One LLM call per review — each is a fully independent unit of work."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content":
             "Classify the customer review. Reply with exactly one word: "
             "POSITIVE, NEGATIVE, or NEUTRAL."},
            {"role": "user", "content": review},
        ],
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print(f"Classifying {len(REVIEWS)} reviews with {MODEL}\n")

    # --- Sequential ---
    print("=== Sequential ===")
    start = time.time()
    for review in REVIEWS:
        label = classify(review)
        print(f"  [{time.time() - start:4.1f}s] {label:8} | {review[:45]}...")
    seq = time.time() - start
    print(f"  Total: {seq:.1f}s\n")

    # --- Parallel ---
    print("=== Parallel (ThreadPoolExecutor) ===")
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(REVIEWS)) as ex:
        futures = {ex.submit(classify, r): r for r in REVIEWS}
        for f in as_completed(futures):
            review = futures[f]
            try:
                label = f.result()
            except Exception as e:
                label = f"ERROR({type(e).__name__})"     # never let one call sink the batch
            print(f"  [{time.time() - start:4.1f}s] {label:8} | {review[:45]}...")
    par = time.time() - start
    print(f"  Total: {par:.1f}s\n")

    print(f"Speedup: {seq / par:.1f}x")
    print(f"Cost: IDENTICAL. Same {len(REVIEWS)} calls, same tokens.")
    print("You bought latency with concurrency, not with money.")
    print()
    print("Sizing max_workers: your OpenAI tier has a requests-per-minute and a")
    print("tokens-per-minute ceiling. Exceed it and you get 429s, which threads")
    print("will hit FASTER. Cap the pool, and pair it with retry/backoff.")
