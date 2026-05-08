"""
Script 2: Confidence scoring — detecting when the model is uncertain.

Idea: work in LOG SPACE to compute confidence.
    1. Sum logprobs of all generated tokens (= log of product of probs)
    2. Divide by N tokens (= log of geometric mean)
    3. Exponent once at the end (= geometric mean probability = confidence)

    High avg logprob (close to 0)  -> model was confident on most tokens
    Low avg logprob (very negative) -> model was guessing on many tokens

Why log space?
    - Multiplying many small probabilities causes underflow (hits 0.0)
    - Adding logprobs stays numerically stable for any sequence length
    - Exponent only once at the end — no intermediate precision loss

Use case in agents:
    - Agent answers a question confidently -> serve the answer
    - Agent is uncertain -> fall back to a search tool, or escalate to human

This script asks 4 questions of varying difficulty and scores confidence.
"""

import os
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_response_with_confidence(question):
    """Ask a question, return the answer + confidence score."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the question in one short sentence."},
            {"role": "user", "content": question}
        ],
        max_tokens=50,
        temperature=0,
        logprobs=True,
        top_logprobs=3,
    )

    content = response.choices[0].message.content
    logprobs_data = response.choices[0].logprobs.content

    # Compute confidence in LOG SPACE:
    #   1. Sum all logprobs (= log of product of all token probs)
    #   2. Divide by N (= log of geometric mean)
    #   3. Exponent at the end to get the geometric mean probability
    # This avoids underflow from multiplying many small probabilities.
    token_details = []
    for t in logprobs_data:
        token_details.append((t.token, t.logprob, math.exp(t.logprob)))

    sum_logprobs = sum(t.logprob for t in logprobs_data)
    avg_logprob = sum_logprobs / len(logprobs_data)
    confidence = math.exp(avg_logprob)  # geometric mean probability

    min_logprob = min(t.logprob for t in logprobs_data)
    min_idx = next(i for i, t in enumerate(logprobs_data) if t.logprob == min_logprob)
    min_token = token_details[min_idx]

    return {
        "question": question,
        "answer": content,
        "confidence": confidence,
        "avg_logprob": avg_logprob,
        "min_prob": math.exp(min_logprob),
        "min_token": min_token,
        "num_tokens": len(logprobs_data),
        "token_details": token_details,
    }


# --- Questions from easy (high confidence) to hard (low confidence) ---
questions = [
    "What is the capital of France?",
    "What is 15 multiplied by 23?",
    "Who won the FIFA World Cup in 2018?",
    "What is the population of Dharamshala, India?",
]

print("=" * 80)
print("CONFIDENCE SCORING: Same model, different questions")
print("=" * 80)

results = []
for q in questions:
    r = get_response_with_confidence(q)
    results.append(r)

for r in results:
    confidence_label = (
        "HIGH" if r["confidence"] > 0.90 else
        "MEDIUM" if r["confidence"] > 0.70 else
        "LOW"
    )

    print(f"\nQ: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"   Avg logprob:      {r['avg_logprob']:.3f}")
    print(f"   Confidence:       {r['confidence']:.1%}  (geometric mean of token probs)")
    print(f"   Lowest token:     {repr(r['min_token'][0])} at {r['min_prob']:.1%}")
    print(f"   Label:            [{confidence_label}]")

# --- Show how this drives agent decisions ---
print("\n" + "=" * 80)
print("HOW AN AGENT WOULD USE THIS")
print("=" * 80)

CONFIDENCE_THRESHOLD = 0.80

for r in results:
    action = (
        "-> Serve answer directly"
        if r["confidence"] > CONFIDENCE_THRESHOLD
        else "-> UNCERTAIN: fall back to search tool or escalate to human"
    )
    print(f"\n  Q: {r['question']}")
    print(f"  Confidence: {r['confidence']:.1%} {action}")

print(f"\n  Threshold: {CONFIDENCE_THRESHOLD:.0%}")
print("  Above threshold -> agent answers autonomously")
print("  Below threshold -> agent uses a tool or asks for help")
