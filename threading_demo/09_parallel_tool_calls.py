"""
Example 9: The agentic payoff — executing parallel tool calls.

This is the reason we learned threads.

When you give a model several tools and ask a question that needs more than
one, it does not ask for them one at a time. It returns a LIST of tool_calls
in a single response:

    message.tool_calls = [get_order(...), get_shipping(...), get_policy(...)]

The model has already told you these are independent. If you then execute them
in a for loop, YOU are the bottleneck — not the model, not the tools.

So: loop to DISPATCH into a pool, then collect. The results go back as
`role: "tool"` messages (one per call, matched by tool_call_id, in any order)
and the model writes the final answer.

Run:  python 09_parallel_tool_calls.py
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)
MODEL = "gpt-4o-mini"


# ---------- the tools (each one waits on a different backend) ----------

def get_order(order_id):
    time.sleep(2)                                  # orders DB
    return {"order_id": order_id, "item": "Standing desk", "total_inr": 24999,
            "placed_on": "2026-07-18"}


def get_shipping(order_id):
    time.sleep(3)                                  # courier API — the slow one
    return {"order_id": order_id, "status": "in_transit", "eta": "2026-08-03",
            "carrier": "BlueDart"}


def get_refund_policy(category):
    time.sleep(1)                                  # docs service
    return {"category": category, "window_days": 30,
            "condition": "unused, original packaging"}


TOOL_IMPLS = {
    "get_order": get_order,
    "get_shipping": get_shipping,
    "get_refund_policy": get_refund_policy,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_order", "description": "Fetch an order record by id.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_shipping", "description": "Fetch live shipping status for an order.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_refund_policy", "description": "Fetch the refund policy for a product category.",
        "parameters": {"type": "object",
                       "properties": {"category": {"type": "string"}},
                       "required": ["category"]}}},
]


def run_tool(tool_call):
    """Executed inside a worker thread. Returns the tool message for the LLM."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    started = time.time()
    try:
        output = TOOL_IMPLS[name](**args)
    except Exception as e:
        output = {"error": f"{type(e).__name__}: {e}"}   # the model can read this and adapt
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,                    # THIS is what pairs it back up
        "content": json.dumps(output),
    }, name, time.time() - started


def answer(question, parallel):
    messages = [
        {"role": "system", "content":
         "You are a customer support agent. Use the tools you need, then answer "
         "in 3 short sentences. Order ids look like ORD-XXXX."},
        {"role": "user", "content": question},
    ]

    # --- Turn 1: the model decides which tools to call ---
    first = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS, temperature=0)
    msg = first.choices[0].message
    messages.append(msg)

    calls = msg.tool_calls or []
    print(f"  Model requested {len(calls)} tool call(s): "
          f"{[c.function.name for c in calls]}")

    # --- Execute them ---
    start = time.time()
    if parallel:
        with ThreadPoolExecutor(max_workers=len(calls) or 1) as ex:
            outcomes = list(ex.map(run_tool, calls))     # dispatch all, then collect
    else:
        outcomes = [run_tool(c) for c in calls]          # one at a time

    for _, name, took in outcomes:
        print(f"    {name:18} {took:.1f}s")
    print(f"  Tool phase: {time.time() - start:.1f}s")

    messages.extend(tool_msg for tool_msg, _, _ in outcomes)

    # --- Turn 2: the model writes the answer from the tool results ---
    second = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0)
    return second.choices[0].message.content


if __name__ == "__main__":
    question = ("For order ORD-7781: what is it, where is it right now, and can "
                "I still return it? It's furniture.")
    print(f"Q: {question}\n")

    print("=== Sequential tool execution ===")
    start = time.time()
    reply = answer(question, parallel=False)
    seq = time.time() - start
    print(f"  Wall clock: {seq:.1f}s\n  A: {reply}\n")

    print("=== Parallel tool execution ===")
    start = time.time()
    reply = answer(question, parallel=True)
    par = time.time() - start
    print(f"  Wall clock: {par:.1f}s\n  A: {reply}\n")

    print(f"End-to-end speedup: {seq / par:.1f}x")
    print()
    print("What changed: ONE line. `[run_tool(c) for c in calls]` became")
    print("`ex.map(run_tool, calls)`. Same tools, same tokens, same answer.")
    print()
    print("Only parallelise calls from the SAME assistant message. The model")
    print("batched them precisely because they don't depend on each other.")
    print("Two tools across two turns may well be sequential by design —")
    print("e.g. search_orders() first, then get_order(id) on what it found.")
