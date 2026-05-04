"""
Step 3: SSE client.
Parses the Server-Sent Events protocol:
  - Reads lines starting with "data: "
  - Strips the prefix, parses JSON payload
  - Stops when it sees "data: [DONE]"

This is the same parsing pattern used by OpenAI's streaming API.
"""

import time
import json
import requests

print("Sending request... (SSE stream)\n")

start = time.time()
response = requests.get("http://localhost:8000/generate/sse", stream=True)

full_text = []

for line in response.iter_lines(decode_unicode=True):
    # SSE sends empty lines as separators — skip them
    if not line:
        continue

    # Every meaningful line starts with "data: "
    if line.startswith("data: "):
        payload = line[len("data: "):]

        # Check for stream-end signal
        if payload == "[DONE]":
            print("\n\n[Stream complete]")
            break

        # Parse the JSON payload
        event = json.loads(payload)
        word = event["word"]
        full_text.append(word)
        print(word, end=" ", flush=True)

print(f"[Total time: {time.time() - start:.1f}s]")
print(f"\nFull text: {' '.join(full_text)}")
