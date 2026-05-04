"""
Script 2: top_k vs top_p vs min_p — three ways to filter the probability distribution.

After temperature rescales the probabilities, we still have 50,000+ tokens
with non-zero probability. Most are garbage. We need to FILTER before sampling.

Three approaches:
  - top_k: keep the K highest-probability tokens, zero out the rest.
  - top_p (nucleus): keep the smallest set of tokens whose cumulative probability >= p.
  - min_p: keep tokens with probability >= (min_p * probability_of_top_token).

This script shows which tokens survive each filter on the SAME logits,
and demonstrates the flaw of top_p that min_p fixes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- Setup ---
MODEL_ID = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device).eval()

prompt = "The future of artificial intelligence is"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    logits = model(input_ids).logits[0, -1, :]

# Use temperature=1.0 (default) for this demo
probs = F.softmax(logits, dim=-1)

# Get top 30 tokens for display
topk_30 = torch.topk(probs, k=30)
top_tokens = [tokenizer.decode([tid]).strip() for tid in topk_30.indices.tolist()]
top_probs = topk_30.values.tolist()


# ===== TOP-K FILTERING =====
K = 5
print(f'Prompt: "{prompt}"\n')
print(f"Total tokens in vocabulary: {len(probs)}")
print(f"Tokens with prob > 0.001: {(probs > 0.001).sum().item()}\n")

print(f"{'='*60}")
print(f"TOP-K (K={K}): Keep the {K} highest-probability tokens")
print(f"{'='*60}")
topk_survivors = top_tokens[:K]
topk_probs = top_probs[:K]
topk_sum = sum(topk_probs)
for i, (tok, p) in enumerate(zip(topk_survivors, topk_probs)):
    print(f"  {i+1}. {repr(tok):<15} {p:.4f} ({p*100:.1f}%)")
print(f"  Cumulative: {topk_sum:.4f} ({topk_sum*100:.1f}%)")
print(f"  Remaining {len(probs)-K} tokens: zeroed out")
print(f"  Problem: K is fixed. Whether model is confident or uncertain,")
print(f"  you always keep exactly {K}. Doesn't adapt.\n")


# ===== TOP-P (NUCLEUS) FILTERING =====
P = 0.90
print(f"{'='*60}")
print(f"TOP-P / NUCLEUS (P={P}): Keep smallest set with cumulative prob >= {P}")
print(f"{'='*60}")

cumulative = 0.0
topp_survivors = []
for tok, p in zip(top_tokens, top_probs):
    topp_survivors.append((tok, p))
    cumulative += p
    if cumulative >= P:
        break

for i, (tok, p) in enumerate(topp_survivors):
    cum_so_far = sum(pp for _, pp in topp_survivors[:i+1])
    print(f"  {i+1}. {repr(tok):<15} {p:.4f} ({p*100:.1f}%)  cumulative: {cum_so_far:.4f}")
print(f"  Tokens kept: {len(topp_survivors)} (adapts to model confidence)")
print(f"  Problem: when model is VERY confident (top token at 80%),")
print(f"  top_p=0.9 still includes low-quality tokens to reach 90%.\n")


# ===== MIN-P FILTERING =====
MIN_P = 0.1
top_token_prob = top_probs[0]
threshold = MIN_P * top_token_prob

print(f"{'='*60}")
print(f"MIN-P (min_p={MIN_P}): Keep tokens with prob >= {MIN_P} * top_prob")
print(f"{'='*60}")
print(f"  Top token prob: {top_token_prob:.4f}")
print(f"  Threshold: {MIN_P} * {top_token_prob:.4f} = {threshold:.4f}\n")

minp_survivors = []
for tok, p in zip(top_tokens, top_probs):
    if p >= threshold:
        minp_survivors.append((tok, p))

for i, (tok, p) in enumerate(minp_survivors):
    print(f"  {i+1}. {repr(tok):<15} {p:.4f} ({p*100:.1f}%)  {'<-- top' if i == 0 else ''}")
print(f"  Tokens kept: {len(minp_survivors)}")
print(f"  Why this is better: threshold SCALES with model confidence.")
print(f"  Confident model (top=0.8) -> threshold=0.08 -> few survivors.")
print(f"  Uncertain model (top=0.05) -> threshold=0.005 -> many survivors.")
print(f"  Adapts both ways. top_p only adapts one way.\n")


# ===== VISUALIZATION =====
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f'Sampling Filters Compared\n'
    f'Prompt: "{prompt}"',
    fontsize=13, fontweight="bold", y=1.02
)

SHOW = 15  # show top 15 tokens in each chart
labels = [t if len(t) <= 10 else t[:8] + ".." for t in top_tokens[:SHOW]]
base_probs = top_probs[:SHOW]

# --- Top-K ---
ax = axes[0]
colors_k = ["#2e86c1" if i < K else "#d5d8dc" for i in range(SHOW)]
ax.bar(range(SHOW), base_probs, color=colors_k, edgecolor="white", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(range(SHOW))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
ax.set_title(f"top_k = {K}", fontsize=12, fontweight="bold", color="#2e86c1")
ax.set_ylabel("Probability", fontsize=10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
ax.text(0.95, 0.92, f"Keep top {K}, zero rest\nFixed — doesn't adapt",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        fontstyle="italic", color="#555")
# Draw cutoff line
ax.axvline(x=K - 0.5, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(K - 0.3, max(base_probs) * 0.8, "cutoff", fontsize=8, color="#e74c3c")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Top-P ---
ax = axes[1]
n_topp = len(topp_survivors)
colors_p = ["#27ae60" if i < n_topp else "#d5d8dc" for i in range(SHOW)]
ax.bar(range(SHOW), base_probs, color=colors_p, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(SHOW))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
ax.set_title(f"top_p = {P}", fontsize=12, fontweight="bold", color="#27ae60")
ax.set_ylabel("Probability", fontsize=10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
cum_at_cut = sum(p for _, p in topp_survivors)
ax.text(0.95, 0.92, f"Keep until cumulative >= {P}\n{n_topp} tokens (cumul: {cum_at_cut:.1%})",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        fontstyle="italic", color="#555")
ax.axvline(x=n_topp - 0.5, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(n_topp - 0.3, max(base_probs) * 0.8, f"cumul={cum_at_cut:.0%}",
        fontsize=8, color="#e74c3c")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Min-P ---
ax = axes[2]
n_minp = len(minp_survivors)
colors_m = ["#e74c3c" if i < n_minp else "#d5d8dc" for i in range(SHOW)]
ax.bar(range(SHOW), base_probs, color=colors_m, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(SHOW))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
ax.set_title(f"min_p = {MIN_P}", fontsize=12, fontweight="bold", color="#e74c3c")
ax.set_ylabel("Probability", fontsize=10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
ax.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(SHOW - 1, threshold + 0.002, f"threshold = {threshold:.4f}",
        fontsize=8, color="#e74c3c", ha="right")
ax.text(0.95, 0.92, f"Keep if prob >= {MIN_P} x top_prob\n{n_minp} tokens (scales with confidence)",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        fontstyle="italic", color="#555")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

output_path = "/Users/vivekanandvivek/Desktop/teaching/Deep Learning and Agentic AI course/Agentic AI Classes/Class-1/Week1_AI_Engineering-1/sampling_deep_dive/sampling_filters.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Chart saved to: {output_path}")
