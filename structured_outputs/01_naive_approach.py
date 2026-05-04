"""
Step 1: The naive approach — ask the model to return JSON in the prompt.

Problem: It works SOMETIMES. But the model is free to:
  - Wrap JSON in markdown backticks (```json ... ```)
  - Add explanatory text before/after the JSON
  - Use different key names ("Name" vs "name" vs "full_name")
  - Return numbers as strings ("25 years" instead of 25)

Run this multiple times — you'll see inconsistent output.
json.loads() will crash on some runs.
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

# Attempt 1: Just ask nicely
print("=== Attempt 1: Polite request ===\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Extract the name, age, city, and company from this text. Return JSON.\n\n{text}"}
    ],
)

raw = response.choices[0].message.content
print(f"Raw response:\n{raw}\n")

try:
    parsed = json.loads(raw)
    print(f"json.loads() succeeded: {parsed}")
except json.JSONDecodeError as e:
    print(f"json.loads() FAILED: {e}")
    print("(The model probably added backticks or extra text around the JSON)")


# Attempt 2: Be more specific in the prompt
print("\n\n=== Attempt 2: Strict instructions ===\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a data extraction assistant. Return ONLY valid JSON with no extra text, no backticks, no markdown."},
        {"role": "user", "content": f'Extract from this text. Keys must be exactly: "name" (string), "age" (integer), "city" (string), "company" (string).\n\n{text}'}
    ],
)

raw = response.choices[0].message.content
print(f"Raw response:\n{raw}\n")

try:
    parsed = json.loads(raw)
    print(f"json.loads() succeeded: {parsed}")
    print(f"Type of age: {type(parsed.get('age'))}")
except json.JSONDecodeError as e:
    print(f"json.loads() FAILED: {e}")

print("\n--- Lesson ---")
print("Even with strict instructions, the model CAN disobey.")
print("Asking nicely is not engineering. We need guarantees.")
