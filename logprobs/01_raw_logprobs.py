"""
Script 1: Raw logprobs — seeing inside the black box.

Without logprobs, the OpenAI API is a black box:
    prompt in -> text out. No visibility into model confidence.

With logprobs=True, you get to peek inside:
    - For each token the model generated: its log-probability
    - The top N alternatives the model considered + their log-probabilities

What is a logprob?
    logprob = log(probability)       # natural log, base e

    logprob     probability     meaning
    -------     -----------     -------
     0.0        100%            impossible (would mean model is 100% sure)
    -0.01        99%            extremely confident
    -0.10        90%            very confident
    -0.69        50%            coin flip
    -2.30        10%            unlikely
    -4.60         1%            very unlikely

    More negative = less confident. Always negative (or zero).
    To convert: probability = e^logprob   (math.exp(logprob))
"""

import os
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "The capital of Japan is"

print(f'Prompt: "{prompt}"\n')

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=30,
    temperature=0,
    logprobs=True,          # <-- enable logprobs
    top_logprobs=5,         # <-- show top 5 alternatives per position
)

content = response.choices[0].message.content
logprobs_data = response.choices[0].logprobs.content

print(f"Response: {content}\n")
print("=" * 80)
print("TOKEN-BY-TOKEN BREAKDOWN")
print("=" * 80)

for i, token_info in enumerate(logprobs_data):
    token = token_info.token
    logprob = token_info.logprob
    prob = math.exp(logprob) * 100  # convert to percentage

    print(f"\nPosition {i}: {repr(token)}")
    print(f"  Chosen token logprob: {logprob:.4f}  ->  probability: {prob:.1f}%")

    # Show alternatives
    print(f"  Top {len(token_info.top_logprobs)} alternatives:")
    for alt in token_info.top_logprobs:
        alt_prob = math.exp(alt.logprob) * 100
        marker = "  <-- chosen" if alt.token == token else ""
        print(f"    {repr(alt.token):<20} logprob: {alt.logprob:>8.4f}  prob: {alt_prob:>6.1f}%{marker}")

print("\n" + "=" * 80)
print("KEY TAKEAWAY")
print("=" * 80)
print("""
logprob = log(probability). More negative = less confident.

The model didn't just output "Tokyo" -- it considered alternatives
and assigned probabilities to each. logprobs lets YOU see those
probabilities and make engineering decisions based on them.
""")
