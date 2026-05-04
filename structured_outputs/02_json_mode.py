"""
Step 2: JSON mode — guaranteed valid JSON, but no schema enforcement.

response_format={"type": "json_object"} tells OpenAI:
  "Your response MUST be valid JSON. No backticks, no extra text."

This solves the parsing problem — json.loads() will ALWAYS work.
But it does NOT control:
  - Which keys appear ("name" vs "full_name" vs "person_name")
  - Value types (age as int vs string)
  - Which fields are present

Run this multiple times and inspect the keys — they may vary.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = """
Rahul Sharma is a 28-year-old software engineer based in Bangalore.
He graduated from IIT Delhi in 2020 and currently works at Flipkart.
He enjoys playing cricket on weekends and is learning to play the guitar.
"""

print("=== JSON Mode ===\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract person details from the text. Return as JSON."},
        {"role": "user", "content": text}
    ],
    response_format={"type": "json_object"},   # <-- the only change
)

raw = response.choices[0].message.content
print(f"Raw response:\n{raw}\n")

# This will ALWAYS succeed now — the API guarantees valid JSON
parsed = json.loads(raw)
print(f"json.loads() succeeded: {parsed}")
print(f"Keys returned: {list(parsed.keys())}")

# But can we trust the keys?
print("\n--- Checking expected keys ---")
expected_keys = ["name", "age", "city", "company"]
for key in expected_keys:
    if key in parsed:
        print(f"  '{key}': {parsed[key]} (type: {type(parsed[key]).__name__})")
    else:
        print(f"  '{key}': MISSING! Model used a different key name.")

print("\n--- Lesson ---")
print("JSON mode guarantees VALID JSON (no more parsing crashes).")
print("But it does NOT guarantee the RIGHT SCHEMA.")
print("The model might use 'full_name' instead of 'name',")
print("or return age as '28 years' instead of 28.")
print("We need schema enforcement — that's Step 3.")
