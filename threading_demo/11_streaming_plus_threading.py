"""
Example 11: The finale — streaming AND threading, together.

Everything in Week 1 converges here. One question, ten tool calls, three
implementations, three stopwatches.

  V1  BLOCKING + SEQUENTIAL
      Wait for the entire LLM response. Then loop over tool_calls and run
      them one at a time.
          cost = full_response + sum(all tools)

  V2  STREAMING + SEQUENTIAL
      Stream the response. The moment ONE tool call has finished arriving,
      run it -- then go back to streaming the rest.
          cost = time_to_first_tool_call + sum(all tools)
      You start working before the model has stopped talking.

      SPOILER, and the most useful thing in this file: this barely helps, and
      the script measures exactly why. V2 can only save the gap between the
      FIRST tool call arriving and the LAST one arriving. The model emits all
      ten in a burst of about 0.2s, so that is the entire prize -- smaller than
      run-to-run API variance. Expect V2 to land within noise of V1, and
      sometimes behind it. Do not "fix" that; it is the finding.

  V3  STREAMING + PARALLEL
      Same detection, but submit() the tool into a pool and immediately go
      back to streaming. Collect every result only at the very end.
          cost ~ max(stream, time_to_first_tool_call + slowest tool)

The interesting part is how little code separates V2 from V3. Both drive the
same generator, stream_tool_calls(), which yields each tool call the moment
its arguments have finished streaming:

      V2:   for call in stream_tool_calls(stream):
                results.append(run_tool(call))          # blocks here

      V3:   for call in stream_tool_calls(stream):
                futures.append(ex.submit(run_tool, call))   # does not block

That generator is the Week 1 lesson in one place: yield turns a stream of
fragments into a stream of complete units, and a thread pool turns a stream of
complete units into work that overlaps.

Run:  python 11_streaming_plus_threading.py
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)
MODEL = "gpt-4o-mini"

T0 = time.time()          # reset before each variant; every log line is relative to it


def log(msg, indent=0):
    """Timestamped log. The timestamps ARE the lesson -- read them, not the text."""
    print(f"  [{time.time() - T0:5.1f}s] {'  ' * indent}{msg}", flush=True)


# ============================================================================
# Ten tools. Each waits on a different backend, so each has its own latency.
# ============================================================================

LATENCY = {
    "get_order":           1.2,
    "get_shipping":        1.8,
    "get_inventory":       0.9,
    "get_refund_policy":   0.7,
    "get_customer_tier":   1.1,
    "get_payment_status":  1.4,
    "get_warranty":        1.0,
    "get_support_tickets": 2.2,     # the slowest — it sets V3's floor
    "get_product_reviews": 1.6,
    "get_delivery_slots":  1.3,
}

RETURNS = {
    "get_order":           {"item": "Standing desk", "total_inr": 24999, "placed": "2026-07-18"},
    "get_shipping":        {"status": "in_transit", "eta": "2026-08-11", "carrier": "BlueDart"},
    "get_inventory":       {"in_stock": 3, "warehouse": "BLR-2"},
    "get_refund_policy":   {"window_days": 15, "note": "furniture: pickup fee Rs.500"},
    "get_customer_tier":   {"tier": "gold", "since": "2023"},
    "get_payment_status":  {"paid": True, "method": "UPI"},
    "get_warranty":        {"years": 3, "covers": "frame and motor"},
    "get_support_tickets": {"open": 2, "last_contact": "2026-08-05"},
    "get_product_reviews": {"avg_rating": 4.3, "count": 812},
    "get_delivery_slots":  {"next_available": "2026-08-12 10:00-13:00"},
}


def make_tool(name):
    def tool(**kwargs):
        time.sleep(LATENCY[name])
        return RETURNS[name]
    return tool


TOOL_IMPLS = {name: make_tool(name) for name in LATENCY}

# Every tool takes order_id so the model can always fill the arguments in.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": name,
        "description": f"Fetch {name.replace('get_', '').replace('_', ' ')} for an order.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}}
    for name in LATENCY
]

SYSTEM = ("You are running a pre-escalation audit for a customer support team. "
          "Before you can answer anything you must gather the complete picture: "
          "call EVERY tool available to you, in one batch. Do not skip any tool "
          "and do not ask the user for permission.")

QUESTION = ("Customer is escalating about order ORD-7781, a delayed standing desk. "
            "Run the full pre-escalation audit.")


def run_tool(call):
    """Executed either inline (V1, V2) or in a worker thread (V3)."""
    name, args = call["name"], call["args"]
    log(f"-> {name} started", indent=1)
    try:
        output = TOOL_IMPLS[name](**json.loads(args or "{}"))
    except Exception as e:
        output = {"error": f"{type(e).__name__}: {e}"}
    log(f"<- {name} done", indent=1)
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(output)}


# ============================================================================
# The generator that makes V2 and V3 possible.
# ============================================================================

def stream_tool_calls(stream):
    """
    Consume an OpenAI stream and yield each tool call the MOMENT it is complete.

    Tool calls arrive as fragments. A single call is spread over many chunks:

        chunk 1  index=0  id='call_abc'  name='get_order'  arguments=''
        chunk 2  index=0                                   arguments='{"order'
        chunk 3  index=0                                   arguments='_id":"ORD'
        ...
        chunk 9  index=1  id='call_def'  name='get_shipping'          <-- index CHANGED

    That index change is the signal: call 0 will never receive another fragment,
    so call 0 is complete and can be yielded RIGHT NOW -- while calls 1..9 are
    still arriving. The last call is yielded when the stream ends.
    """
    pending = {}
    open_index = None

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            continue

        for fragment in delta.tool_calls:
            idx = fragment.index

            if idx != open_index:
                if open_index is not None:
                    yield pending[open_index]       # the previous one is now final
                open_index = idx
                pending[idx] = {"id": "", "name": "", "args": ""}

            if fragment.id:
                pending[idx]["id"] += fragment.id
            if fragment.function and fragment.function.name:
                pending[idx]["name"] += fragment.function.name
            if fragment.function and fragment.function.arguments:
                pending[idx]["args"] += fragment.function.arguments

    if open_index is not None:
        yield pending[open_index]                   # the last one


def assistant_message(calls):
    """Rebuild the assistant turn from streamed fragments, for the follow-up call."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]}}
                           for c in calls]}


def final_answer(messages):
    """Turn 2 -- identical work in all three variants, so it cancels out."""
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    return resp.choices[0].message.content


# ============================================================================
# V1 — blocking call, sequential tools
# ============================================================================

def v1_blocking_sequential():
    global T0
    T0 = time.time()
    print("\n" + "=" * 78)
    print("V1  BLOCKING + SEQUENTIAL   (no streaming, no threading)")
    print("=" * 78)

    log("requesting completion (stream=False) -- nothing to do but wait")
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": QUESTION}],
        tools=TOOL_SCHEMAS, temperature=0)
    msg = resp.choices[0].message
    llm_done = time.time() - T0

    calls = [{"id": tc.id, "name": tc.function.name, "args": tc.function.arguments}
             for tc in (msg.tool_calls or [])]
    log(f"response complete: {len(calls)} tool calls received AT ONCE")
    log("now running them one after another")

    tool_start = time.time()
    tool_messages = [run_tool(c) for c in calls]        # <-- sequential
    tool_phase = time.time() - tool_start

    log("all tools done, asking for the final answer")
    answer = final_answer([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": QUESTION},
                           assistant_message(calls), *tool_messages])
    total = time.time() - T0
    log("final answer received")
    return {"variant": "V1  blocking + sequential", "n": len(calls),
            "llm": llm_done, "tools": tool_phase, "total": total, "answer": answer}


# ============================================================================
# V2 — streamed call, tools run inline as they are detected
# ============================================================================

def v2_streaming_sequential():
    global T0
    T0 = time.time()
    print("\n" + "=" * 78)
    print("V2  STREAMING + SEQUENTIAL   (detect early, but still one at a time)")
    print("=" * 78)

    log("opening stream (stream=True)")
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": QUESTION}],
        tools=TOOL_SCHEMAS, temperature=0, stream=True)

    calls, tool_messages = [], []
    first_detect = None

    for call in stream_tool_calls(stream):
        if first_detect is None:
            first_detect = time.time() - T0
            log(f"FIRST tool call detected mid-stream: {call['name']}")
        calls.append(call)
        tool_messages.append(run_tool(call))            # <-- blocks the stream
    stream_done = time.time() - T0

    log(f"stream finished, {len(calls)} tool calls, all already executed")
    answer = final_answer([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": QUESTION},
                           assistant_message(calls), *tool_messages])
    total = time.time() - T0
    log("final answer received")
    return {"variant": "V2  streaming + sequential", "n": len(calls),
            "llm": first_detect, "tools": stream_done - first_detect,
            "total": total, "answer": answer}


# ============================================================================
# V3 — streamed call, tools dispatched to a pool and collected at the end
# ============================================================================

def v3_streaming_parallel():
    global T0
    T0 = time.time()
    print("\n" + "=" * 78)
    print("V3  STREAMING + PARALLEL   (detect early AND never wait)")
    print("=" * 78)

    log("opening stream (stream=True)")
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": QUESTION}],
        tools=TOOL_SCHEMAS, temperature=0, stream=True)

    calls, futures = [], []
    first_detect = last_detect = None

    with ThreadPoolExecutor(max_workers=len(LATENCY)) as ex:
        for call in stream_tool_calls(stream):
            if first_detect is None:
                first_detect = time.time() - T0
                log(f"FIRST tool call detected mid-stream: {call['name']}")
            last_detect = time.time() - T0             # nothing blocks here, so this
            calls.append(call)                         # measures true arrival timing
            futures.append(ex.submit(run_tool, call))  # <-- fire and keep streaming
        stream_done = time.time() - T0
        log(f"stream finished at {stream_done:.1f}s -- tools still running, collecting")

        tool_messages = []
        for f in futures:                               # input order = tool_call order
            try:
                tool_messages.append(f.result())
            except Exception as e:
                log(f"tool failed: {e}")

    collected = time.time() - T0
    log("all tools collected")
    answer = final_answer([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": QUESTION},
                           assistant_message(calls), *tool_messages])
    total = time.time() - T0
    log("final answer received")
    return {"variant": "V3  streaming + parallel", "n": len(calls),
            "llm": first_detect, "tools": collected - first_detect,
            "total": total, "answer": answer,
            "spread": last_detect - first_detect}      # <-- the ceiling on V2's win


# ============================================================================

if __name__ == "__main__":
    print(f"Question: {QUESTION}")
    print(f"Tools available: {len(LATENCY)}   "
          f"sum of latencies: {sum(LATENCY.values()):.1f}s   "
          f"slowest: {max(LATENCY.values()):.1f}s")

    runs = [v1_blocking_sequential(), v2_streaming_sequential(), v3_streaming_parallel()]

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"{'variant':30} {'tools':>6} {'to 1st':>8} {'tool phase':>11} {'TOTAL':>8}")
    print("-" * 78)
    for r in runs:
        print(f"{r['variant']:30} {r['n']:>6} {r['llm']:>7.1f}s "
              f"{r['tools']:>10.1f}s {r['total']:>7.1f}s")

    base = runs[0]["total"]
    print()
    for r in runs[1:]:
        delta = base - r["total"]
        verdict = f"{delta:.1f}s faster" if delta >= 0 else f"{-delta:.1f}s SLOWER"
        print(f"  {r['variant']:30} vs V1: {verdict}")

    spread = runs[-1]["spread"]
    print()
    print("Read the phase columns, not just the total:")
    print("  'to 1st'     when the first tool call became executable.")
    print("  'tool phase' V1 and V2 both pay the SUM of all tool latencies "
          f"({sum(LATENCY.values()):.1f}s).")
    print(f"               V3 pays roughly the slowest one ({max(LATENCY.values()):.1f}s).")
    print()
    print("=" * 78)
    print("WHY V2 BARELY HELPS -- the measurement that explains it")
    print("=" * 78)
    print(f"  All 10 tool calls arrived within {spread:.2f}s of each other.")
    print()
    print("  V2's entire advantage is starting tool #1 before tool #10 has")
    print(f"  finished arriving. That window IS the arrival spread: {spread:.2f}s.")
    print("  It cannot save more than that, and run-to-run API variance is")
    print("  around +/-1s -- so the saving is invisible. Expect V2 to land")
    print("  within noise of V1, sometimes slower. That is not a bug.")
    print()
    print("  The reason is how the model emits tool calls: not spread out")
    print("  across the response, but as one burst once it has decided.")
    print("  There is no long tail to overlap with.")
    print()
    observed = runs[0]["total"] - runs[1]["total"]
    excess = abs(observed) - spread
    print(f"  This run V2's observed delta vs V1 was {observed:+.1f}s, "
          f"against a ceiling of {spread:.2f}s.")
    if excess > 0:
        print(f"  So ~{excess:.1f}s of that is just the LLM calls differing "
              "between runs --")
        print("  it is not attributable to your code at all.")
    else:
        print("  That fits inside the ceiling: consistent with a real saving, "
              "but too")
        print("  small to distinguish from noise on a single run.")
    print("  Run the file three times: this number swings both ways, "
          "and V2 will")
    print("  sometimes come out SLOWER than V1. V3 never does.")
    print()
    print("  MEASURE THE PHASE, NOT THE TOTAL. 'tool phase' is the honest")
    print("  column -- it contains no API variance at all:")
    for r in runs:
        print(f"      {r['variant']:30} {r['tools']:>5.1f}s")
    print()
    print("=" * 78)
    print("THE ACTUAL LESSON")
    print("=" * 78)
    print("  Streaming and threading fix DIFFERENT costs, and only one of them")
    print("  is on the critical path here:")
    print()
    print("    streaming  -> shrinks time-to-first-VISIBLE-TOKEN.")
    print("                  Worth everything for text the user reads.")
    print(f"                  Worth {spread:.2f}s for dispatching tool calls.")
    print("    threading  -> shrinks the tool phase from SUM to MAX.")
    print(f"                  {sum(LATENCY.values()):.1f}s -> {max(LATENCY.values()):.1f}s. "
          "That is the whole win.")
    print()
    print("  So do not reach for streaming to make your agent faster. Reach for")
    print("  it to make your agent FEEL responsive, and reach for threads to")
    print("  make it actually be fast. V3 is not a third technique -- it is V2")
    print("  with submit() in place of a direct call.")
    print()
    print("  One production note on V2's design: blocking a live HTTP stream for")
    print(f"  {sum(LATENCY.values()):.1f}s of tool work holds that connection open the whole time.")
    print("  V3 drains the stream in ~3s and does the waiting off to the side.")
    print()
    print(f"Final answer (V3):\n{runs[-1]['answer']}")
