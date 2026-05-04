"""
Script 1: What does temperature ACTUALLY do?

Temperature rescales the logits before softmax:
    probs = softmax(logits / temperature)

- Low temperature (0.1)  -> sharpens the distribution -> model becomes very confident
                            in top token, everything else near zero. Almost argmax.
- Temperature = 1.0      -> default. Probabilities as the model learned them.
- High temperature (1.5) -> flattens the distribution -> more tokens get a fair chance,
                            output becomes more "creative" (and more random/risky).

This script:
  1. Runs GPT-2 on a prompt, grabs the raw logits at the last position
  2. Applies softmax at 4 different temperatures
  3. Prints a terminal comparison table
  4. Generates a publication-quality matplotlib chart showing all 4 distributions
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers / CLI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
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

# --- Get raw logits for the LAST position (next-token prediction) ---
with torch.no_grad():
    logits = model(input_ids).logits[0, -1, :]   # [vocab_size]

# --- Apply softmax at different temperatures ---
temperatures = [0.1, 0.5, 1.0, 1.5]
TOP_K = 20   # show top 20 tokens for each temperature

results = {}
for temp in temperatures:
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)
    topk = torch.topk(probs, k=TOP_K)
    tokens = [tokenizer.decode([tid]).strip() for tid in topk.indices.tolist()]
    probabilities = topk.values.tolist()
    results[temp] = {"tokens": tokens, "probs": probabilities}


# --- Terminal output ---
print(f'Prompt: "{prompt}"\n')
print(f"{'Token':<15}", end="")
for temp in temperatures:
    print(f"{'T=' + str(temp):>12}", end="")
print("\n" + "-" * (15 + 12 * len(temperatures)))

# Use the token list from T=1.0 as the reference order
ref_tokens = results[1.0]["tokens"]
for i, token in enumerate(ref_tokens[:15]):
    print(f"{repr(token):<15}", end="")
    for temp in temperatures:
        # Find this token's probability at this temperature
        if token in results[temp]["tokens"]:
            idx = results[temp]["tokens"].index(token)
            prob = results[temp]["probs"][idx]
        else:
            prob = 0.0
        print(f"{prob:>11.4f}%", end="")  # not actually %, just for readability
    print()

print(f"\nTop token probability at each temperature:")
for temp in temperatures:
    top_prob = results[temp]["probs"][0]
    top_token = results[temp]["tokens"][0]
    print(f"  T={temp}: {repr(top_token)} at {top_prob:.4f} ({top_prob*100:.1f}%)")


# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Effect of Temperature on Next-Token Probabilities\n'
    f'Prompt: "{prompt}"',
    fontsize=14, fontweight="bold", y=0.98
)

# Color scheme: cold (blue) for low temp, warm (red) for high temp
colors = ["#1a5276", "#2e86c1", "#27ae60", "#e74c3c"]
descriptions = [
    "Near-argmax: model is almost certain",
    "Slightly smoothed: top tokens get most mass",
    "Default: probabilities as the model learned",
    "Flattened: many tokens get a fair chance"
]

for idx, (temp, ax) in enumerate(zip(temperatures, axes.flat)):
    tokens = results[temp]["tokens"]
    probs = results[temp]["probs"]

    # Shorten token labels for display
    labels = [t if len(t) <= 12 else t[:10] + ".." for t in tokens]

    bars = ax.bar(range(len(probs)), probs, color=colors[idx], alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    # Highlight the top token
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(1.5)

    # Annotate top token probability
    ax.annotate(
        f"{probs[0]*100:.1f}%",
        xy=(0, probs[0]),
        xytext=(2.5, probs[0] * 1.05 + 0.01),
        fontsize=10, fontweight="bold", color=colors[idx],
        arrowprops=dict(arrowstyle="->", color=colors[idx], lw=1.2)
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(f"Temperature = {temp}", fontsize=12, fontweight="bold",
                 color=colors[idx])
    ax.set_ylim(0, max(probs[0] * 1.25, 0.1))

    # Add description subtitle
    ax.text(0.98, 0.92, descriptions[idx],
            transform=ax.transAxes, fontsize=8, fontstyle="italic",
            ha="right", va="top", color="#555555")

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

plt.tight_layout(rect=[0, 0, 1, 0.93])

output_path = "/Users/vivekanandvivek/Desktop/teaching/Deep Learning and Agentic AI course/Agentic AI Classes/Class-1/Week1_AI_Engineering-1/sampling_deep_dive/temperature_effect.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nChart saved to: {output_path}")
print("(Open the PNG to see the visualization)")
