"""
Step 1: Normal client.
Makes a request, waits for the full response, then prints.
Notice the long pause before anything appears.
"""

import time
import requests

print("Sending request... (you'll wait ~10 seconds with no output)\n")

start = time.time()
response = requests.get("http://localhost:8000/generate")
elapsed = time.time() - start

print(f"[Received after {elapsed:.1f}s]")
print(response.json()["text"])
