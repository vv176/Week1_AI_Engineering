"""
Script 1: Stage 1 — Pre-training.

What is pre-training?
  The model reads billions of tokens from the internet and learns ONE task:
  predict the next token. That's it. No instructions, no Q&A format,
  no concept of "being helpful." Just: given these tokens, what comes next?

What you get: a model that KNOWS everything but FOLLOWS nothing.
  It can complete "The Eiffel Tower is located in" → "Paris"
  But ask "What is the capital of France?" and it might continue with
  more questions, a Wikipedia-style paragraph, or random text.

This script demonstrates that behavior using GPT-2 — a real base model
you can run locally. GPT-2 is a pure Stage 1 model (no SFT, no RLHF).
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("Loading GPT-2 (a pure base model — pre-training only, no fine-tuning)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def generate(prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """Generate text token-by-token using manual sampling loop.

    We avoid model.generate() and instead do what it does under the hood:
    forward pass → logits → sample → append → repeat.
    This is literally how all LLM inference works.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_logits = outputs.logits[0, -1, :]  # logits for last position

            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-k filtering
            top_vals, top_idx = torch.topk(next_logits, top_k)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(0, top_idx, top_vals)

            # Sample from distribution
            probs = F.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, next_token], dim=-1)

    new_tokens = generated[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ===== TEST 1: Completion (what pre-training is good at) =====
print("\n" + "=" * 70)
print("TEST 1: Text completion — what pre-training excels at")
print("=" * 70)

completions = [
    "The Eiffel Tower is located in",
    "India's largest state by area is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The Python programming language was created by",
]

for prompt in completions:
    result = generate(prompt, max_new_tokens=30)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  GPT-2:  {result.strip()!r}")

print("\n  ^ The model COMPLETES text naturally. Pre-training works.")


# ===== TEST 2: Questions (what pre-training is BAD at) =====
print(f"\n\n{'='*70}")
print("TEST 2: Direct questions — where base models fail")
print("=" * 70)

questions = [
    "What is the capital of France?",
    "How do I sort a list in Python?",
    "Explain machine learning in simple terms.",
    "What should I see in Goa?",
]

for prompt in questions:
    result = generate(prompt, max_new_tokens=40)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  GPT-2:  {result.strip()!r}")

print("""
  ^ Notice: the base model doesn't ANSWER questions.
  It CONTINUES text. It might:
  - Add more questions (it's seen Q&A lists on the internet)
  - Write a Wikipedia-style paragraph starting from the question
  - Go off on a tangent

  It KNOWS the answer to "capital of France" (it's seen it millions
  of times in training). But nobody taught it to FORMAT a response
  as a helpful answer. That's what Stage 2 (SFT) fixes.
""")


# ===== TEST 3: Instruction following (completely absent) =====
print("=" * 70)
print("TEST 3: Instructions — base models don't follow them")
print("=" * 70)

instructions = [
    "Translate the following to Hindi: Good morning, how are you?",
    "Summarize in one sentence: Machine learning is a subset of AI that enables systems to learn from data.",
    "List 3 benefits of exercise:",
]

for prompt in instructions:
    result = generate(prompt, max_new_tokens=40)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  GPT-2:  {result.strip()!r}")

print("""
  ^ The model doesn't understand that "Translate", "Summarize",
  or "List" are INSTRUCTIONS. It treats them as text to complete.

  Pre-training gave GPT-2:
    ✓ Language understanding
    ✓ World knowledge (facts, code, reasoning patterns)
    ✓ Grammar and coherence
    ✗ Instruction following
    ✗ Conversational behavior
    ✗ Safety / refusal behavior

  It's a knowledge engine with no manners.
  Stage 2 (SFT) teaches it manners. Stage 3 (RLHF) teaches it judgment.
""")
