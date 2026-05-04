"""
Script 3: The Engineer's Rule — temperature=0 for agents, higher for creativity.

Same prompt, same model, 5 runs each at temperature=0 and temperature=1.0.

temperature=0 (argmax):
  - Picks the highest-probability token every time
  - Output is IDENTICAL across runs — deterministic
  - This is what you want for agents: tool calls must be consistent

temperature=1.0 (sampling):
  - Samples from the full distribution
  - Output is DIFFERENT every run — creative/unpredictable
  - Good for writing assistants, brainstorming, creative tasks

The logits are deterministic (same input -> same logits every time).
Temperature only changes what happens AFTER — how we pick from those fixed probabilities.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "List 3 benefits of learning Python."

NUM_RUNS = 5


def generate(temperature, run_num):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


# --- Temperature = 0 ---
print(f'Prompt: "{prompt}"')
print(f"\n{'='*70}")
print(f"TEMPERATURE = 0 (deterministic / argmax)")
print(f"{'='*70}\n")

results_0 = []
for i in range(NUM_RUNS):
    text = generate(temperature=0, run_num=i)
    results_0.append(text)
    print(f"--- Run {i+1} ---")
    print(text)
    print()

# Check if all runs are identical
all_same = all(r == results_0[0] for r in results_0)
print(f"All {NUM_RUNS} runs identical? {all_same}")
if all_same:
    print("(Same logits -> same argmax -> same output. Every time.)")
else:
    print("(Minor variation possible due to GPU float rounding, but extremely rare.)")


# --- Temperature = 1.0 ---
print(f"\n{'='*70}")
print(f"TEMPERATURE = 1.0 (sampling / creative)")
print(f"{'='*70}\n")

results_1 = []
for i in range(NUM_RUNS):
    text = generate(temperature=1.0, run_num=i)
    results_1.append(text)
    print(f"--- Run {i+1} ---")
    print(text)
    print()

all_same_1 = all(r == results_1[0] for r in results_1)
print(f"All {NUM_RUNS} runs identical? {all_same_1}")
if not all_same_1:
    print("(Same logits, but sampling rolls a different dice each time.)")


# --- The engineering insight ---
print(f"\n{'='*70}")
print("THE ENGINEER'S RULE")
print(f"{'='*70}")
print("""
For AGENTIC use cases (tool calls, structured decisions, routing):
  -> temperature = 0 (or very low, 0.1)
  -> You need the agent to make the SAME decision given the SAME input
  -> A tool call must be deterministic, not creative

For CREATIVE use cases (writing, brainstorming, conversation):
  -> temperature = 0.7 - 1.0
  -> You WANT variety and surprise
  -> Same prompt should give different interesting answers

Remember: the MODEL is deterministic (same input -> same logits).
Temperature controls what happens AFTER — the sampling step.
temperature=0 means "skip the dice roll, just pick the best."
""")
