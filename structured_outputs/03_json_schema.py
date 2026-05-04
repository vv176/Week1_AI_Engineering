"""
Step 3: Structured outputs with JSON schema — exact shape guaranteed.

response_format={"type": "json_schema", "json_schema": {...}} tells OpenAI:
  "Your response MUST be valid JSON matching THIS EXACT SCHEMA."

The model is now CONSTRAINED:
  - Exact key names (what you define is what you get)
  - Exact types (age WILL be an integer, not a string)
  - Required fields WILL be present
  - No extra keys, no surprises

This is the mechanism behind tool calling. When you define a tool's
parameter schema, the model is constrained the same way.
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

# Define the exact schema we want
person_schema = {
    "name": "person_info",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person"
            },
            "age": {
                "type": "integer",
                "description": "Age in years"
            },
            "city": {
                "type": "string",
                "description": "City of residence"
            },
            "company": {
                "type": "string",
                "description": "Current employer"
            },
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of hobbies mentioned"
            }
        },
        "required": ["name", "age", "city", "company", "hobbies"],
        "additionalProperties": False
    }
}

print("=== Structured Output (JSON Schema) ===\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract person details from the text."},
        {"role": "user", "content": text}
    ],
    response_format={"type": "json_schema", "json_schema": person_schema},
)

raw = response.choices[0].message.content
parsed = json.loads(raw)

print(f"Response:\n{json.dumps(parsed, indent=2)}\n")

# Verify exact schema compliance
print("--- Schema verification ---")
print(f"  name:    {parsed['name']!r}  (type: {type(parsed['name']).__name__})")
print(f"  age:     {parsed['age']!r}  (type: {type(parsed['age']).__name__})")
print(f"  city:    {parsed['city']!r}  (type: {type(parsed['city']).__name__})")
print(f"  company: {parsed['company']!r}  (type: {type(parsed['company']).__name__})")
print(f"  hobbies: {parsed['hobbies']!r}  (type: {type(parsed['hobbies']).__name__})")

# Show that we can use the data directly in code — no parsing, no guessing
print("\n--- Using extracted data in code (zero parsing needed) ---")
print(f"{parsed['name']} is {parsed['age']} years old, works at {parsed['company']}.")
print(f"Lives in {parsed['city']}. Hobbies: {', '.join(parsed['hobbies'])}.")

print("\n--- Lesson ---")
print("The model HAD to return exactly these keys, exactly these types.")
print("age is an integer (not '28 years'), hobbies is a list (not a comma-separated string).")
print("This is the same mechanism behind tool calling —")
print("you define a function's parameter schema, the model is constrained to match it.")
print("No regex. No string parsing. No hoping. Just data.")
