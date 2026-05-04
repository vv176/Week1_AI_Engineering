"""
Step 5: OpenAI normal (non-streaming) client.
Compare with client_openai_stream.py — same prompt, same model, but no streaming.
You wait for the ENTIRE response to be generated before seeing anything.
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
print("Waiting for full response...\n")

start = time.time()

# No stream=True — blocks until the entire response is generated
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

elapsed = time.time() - start

# The complete response arrives all at once
content = response.choices[0].message.content
print(content)

print(f"\n[Done in {elapsed:.1f}s]")
print(f"[Tokens used: {response.usage.prompt_tokens} input + {response.usage.completion_tokens} output = {response.usage.total_tokens} total]")
