"""
Script 2: Stage 2 — Supervised Fine-Tuning (SFT).

What is SFT?
  Take the pre-trained base model and train it on ~100K examples of:
      (instruction, ideal_response)
  Human annotators write these. For example:
      instruction: "What is the capital of France?"
      response:    "The capital of France is Paris."

  The model learns: when a human asks a question, I should ANSWER it
  (not just continue the text).

What you get: instruction-following model.
  Same knowledge as the base model, but now it formats its output
  as helpful responses instead of raw text completion.

This script compares GPT-2 (pure base model) vs DialoGPT (SFT'd for
conversation) on the same prompts, then shows what SFT training data
looks like and how it changes behavior.
"""

import torch
import torch.nn.functional as F
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
)

# Load both models
print("Loading GPT-2 (base model — Stage 1 only)...")
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

print("Loading DialoGPT (SFT'd for conversation — Stage 1 + Stage 2)...")
dial_tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dial_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
dial_model.eval()


def _manual_generate(model, tok, input_ids, max_new_tokens=40, temperature=0.8, top_k=50):
    """Manual token-by-token generation (avoids model.generate() bug on macOS + torch 2.9)."""
    generated = input_ids.clone()
    eos_id = tok.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_logits = outputs.logits[0, -1, :] / temperature

            top_vals, top_idx = torch.topk(next_logits, top_k)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(0, top_idx, top_vals)

            probs = F.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)

            if next_token.item() == eos_id:
                break
            generated = torch.cat([generated, next_token], dim=-1)

    new_tokens = generated[0][input_ids.shape[-1]:]
    if len(new_tokens) == 0:
        return "(no response)"
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def generate_gpt2(prompt, max_new_tokens=40):
    input_ids = gpt2_tok.encode(prompt, return_tensors="pt")
    return _manual_generate(gpt2_model, gpt2_tok, input_ids, max_new_tokens)


def generate_dialogpt(prompt, max_new_tokens=40):
    # DialoGPT expects the user turn followed by EOS token
    input_ids = dial_tok.encode(prompt + dial_tok.eos_token, return_tensors="pt")
    return _manual_generate(dial_model, dial_tok, input_ids, max_new_tokens)


# ===== HEAD-TO-HEAD COMPARISON =====
print("\n" + "=" * 70)
print("HEAD-TO-HEAD: Base Model (GPT-2) vs SFT Model (DialoGPT)")
print("=" * 70)

prompts = [
    "Do you like football?",
    "What is your favorite food?",
    "Tell me a joke.",
    "How are you today?",
    "What is the meaning of life?",
]

for prompt in prompts:
    gpt2_out = generate_gpt2(prompt)
    dial_out = generate_dialogpt(prompt)
    print(f"\n  Prompt:    {prompt!r}")
    print(f"  GPT-2:     {gpt2_out!r}")
    print(f"  DialoGPT:  {dial_out!r}")

print("""
  ^ Same underlying architecture (GPT-2 medium). Same pre-training knowledge.
  But DialoGPT was fine-tuned on Reddit conversations — millions of
  (user_message, reply) pairs. That's SFT.

  It learned: when someone says something, I should REPLY to it.
  GPT-2 just continues the text.
""")


# ===== WHAT SFT TRAINING DATA LOOKS LIKE =====
print("=" * 70)
print("WHAT SFT TRAINING DATA LOOKS LIKE")
print("=" * 70)

sft_examples = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.",
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "response": "def reverse_string(s):\n    return s[::-1]",
    },
    {
        "instruction": "Translate to Hindi: Good morning",
        "response": "Good morning in Hindi is: सुप्रभात (Suprabhat)",
    },
    {
        "instruction": "Summarize: Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed.",
        "response": "Machine learning lets computers learn patterns from data automatically, without manual programming.",
    },
]

print("\n  SFT training = thousands of these (instruction, response) pairs:\n")
for i, ex in enumerate(sft_examples, 1):
    print(f"  Example {i}:")
    print(f"    Instruction: {ex['instruction']}")
    print(f"    Response:    {ex['response']}")
    print()

print("""  The model is trained to MINIMIZE the loss on predicting the response
  tokens, given the instruction. Same next-token prediction as
  pre-training, but now the training data is curated (instruction, response)
  pairs instead of raw internet text.

  This is also called "instruction tuning" or "chat fine-tuning."
""")


# ===== WHAT SFT GIVES AND DOESN'T GIVE =====
print("=" * 70)
print("WHAT SFT ADDS vs WHAT'S STILL MISSING")
print("=" * 70)

print("""
  After SFT, the model gains:
    ✓ Instruction following ("translate this", "summarize that")
    ✓ Conversational format (answers questions instead of completing text)
    ✓ Task awareness (knows what "list 3 things" means)

  But SFT alone does NOT give:
    ✗ Judgment about WHICH response is better (just mimics training examples)
    ✗ Safety refusals ("how to hack a server" — SFT model may just answer)
    ✗ Helpfulness vs harmlessness tradeoff (no preference learning)
    ✗ Reducing hallucination (SFT model confidently makes things up)

  SFT teaches the model to FOLLOW instructions.
  RLHF (Stage 3) teaches it WHICH responses humans actually prefer.
""")


# ===== ENGINEER'S RELEVANCE =====
print("=" * 70)
print("WHY THIS MATTERS FOR AI ENGINEERS")
print("=" * 70)

print("""
  When you fine-tune a model for your use case (Module 10 in this course),
  you are essentially RE-DOING Stage 2 on your own data.

  Examples:
    - Fine-tune on (customer_query, ideal_agent_response) pairs
      → model learns your company's tone and format
    - Fine-tune on (code_snippet, code_review) pairs
      → model learns to review code like your team does
    - Fine-tune on (medical_question, doctor_answer) pairs
      → model learns to answer in clinical style

  You are not teaching the model new knowledge (that's pre-training).
  You are teaching it a new BEHAVIOR — how to format its existing
  knowledge for your specific task.

  This is exactly what SFT does: same knowledge, new behavior.
""")
