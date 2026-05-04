"""
Step 4: OpenAI streaming client.
Same pattern as Step 3 — the LLM is just another server sending SSE chunks.
Each chunk contains a "delta" with partial content (a token or few characters).
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "Explain in 10 sentences why the sky is blue."

print(f"Prompt: {prompt}")
print("=" * 50)
print("Streaming response:\n")

start = time.time()

# stream=True gives us an iterator of chunks — same concept as our SSE demo
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

full_response = []

for chunk in stream:
    # Each chunk has a "delta" — the new piece of text (like one word from our server)
    delta = chunk.choices[0].delta
    if delta.content:
        token = delta.content
        full_response.append(token)
        print(token, end="", flush=True)

elapsed = time.time() - start
print(f"\n\n[Done in {elapsed:.1f}s]")
print(f"[Tokens received: {len(full_response)}]")
