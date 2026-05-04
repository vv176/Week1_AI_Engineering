"""
Example 7: Connecting yield to streaming — the bridge.

This shows WHY FastAPI's StreamingResponse needs a generator:
  - The generator yields one piece at a time
  - StreamingResponse calls next() internally
  - Each yielded piece is sent over HTTP immediately
  - The client receives pieces progressively

We simulate the server-side flow WITHOUT FastAPI to show the mechanism.
"""

import time


def simulate_llm_generating(prompt):
    """
    Pretend to be an LLM generating a response token by token.
    Each yield = one token produced.
    """
    response_tokens = ["The", " sky", " is", " blue", " because", " of",
                       " Ray", "leigh", " scattering", "."]
    for token in response_tokens:
        time.sleep(0.3)            # simulate computation time per token
        yield token


def simulate_non_streaming_server(prompt):
    """What a non-streaming server does: buffer everything, send at once."""
    full_response = ""
    for token in simulate_llm_generating(prompt):
        full_response += token     # buffer in memory
    return full_response           # send only after ALL tokens are ready


def simulate_streaming_server(prompt):
    """What a streaming server does: forward each yield to the client."""
    for token in simulate_llm_generating(prompt):
        yield token                # forward immediately — no buffering


if __name__ == "__main__":
    prompt = "Why is the sky blue?"

    # --- Non-streaming: client waits, then gets everything ---
    print("=== Non-streaming (like server_normal.py) ===")
    print("Client waiting...", end="", flush=True)
    start = time.time()
    response = simulate_non_streaming_server(prompt)
    elapsed = time.time() - start
    print(f" [{elapsed:.1f}s wait]")
    print(f"Response: {response}\n")

    # --- Streaming: client sees tokens as they arrive ---
    print("Client receiving: ", end="", flush=True)
    start = time.time()
    for token in simulate_streaming_server(prompt):
        print(token, end="", flush=True)     # print each token as it arrives
    elapsed = time.time() - start
    print(f"  [{elapsed:.1f}s total]")

    print()
    print("Same total time. But the client saw the FIRST token in 0.3s,")
    print("not after 3s. That's the entire point of streaming + yield.")
