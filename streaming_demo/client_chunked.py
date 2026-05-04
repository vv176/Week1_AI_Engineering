"""
Step 2: Chunked streaming client.
Uses requests with stream=True to read bytes as they arrive.
Words appear one by one in the terminal — no waiting for the full response.
"""

import time
import requests

print("Sending request... (words will appear one by one)\n")

start = time.time()
response = requests.get("http://localhost:8000/generate/stream", stream=True)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    # Each chunk is a word (or a few bytes) as yielded by the server
    elapsed = time.time() - start
    print(f"[{elapsed:.1f}s] {chunk}", end="", flush=True)

print(f"\n\n[Done in {time.time() - start:.1f}s]")
