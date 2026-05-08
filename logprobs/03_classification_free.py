"""
Script 3: Classification without fine-tuning — using logprobs as a free classifier.

Instead of parsing the model's text response ("Yes" / "No"),
read the logprobs of the FIRST token to get a calibrated probability.

Prompt: "Is this email spam? Answer exactly Yes or No."
max_tokens=1, logprobs=True

The model outputs one token. We read its logprob AND the logprob
of the alternative. Now we have:
    P(Yes) = 0.95, P(No) = 0.05  ->  "This is almost certainly spam"
    P(Yes) = 0.52, P(No) = 0.48  ->  "This is ambiguous, needs human review"

No fine-tuning. No classifier training. No text parsing.
Just read the probabilities the model already computed.
"""

import os
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_email(email_text):
    """Classify email as spam/not-spam using logprobs."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an email spam classifier. Answer exactly Yes or No."},
            {"role": "user", "content": f"Is this email spam?\n\n{email_text}"}
        ],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )

    token_info = response.choices[0].logprobs.content[0]
    chosen_token = token_info.token

    # Extract probabilities for Yes and No from top_logprobs
    # Accumulate: "Yes", "YES", "yes" all count toward p_yes
    p_yes = 0.0
    p_no = 0.0
    for alt in token_info.top_logprobs:
        clean = alt.token.strip().lower()
        if clean.startswith("yes"):
            p_yes += math.exp(alt.logprob)
        elif clean.startswith("no"):
            p_no += math.exp(alt.logprob)

    return {
        "chosen": chosen_token.strip(),
        "p_yes": p_yes,
        "p_no": p_no,
        "all_alternatives": [(alt.token, math.exp(alt.logprob)) for alt in token_info.top_logprobs],
    }


# --- Test emails ---
emails = [
    {
        "label": "Obvious spam",
        "text": "CONGRATULATIONS!!! You have been selected as the WINNER of our $5,000,000 lottery! Click here NOW to claim your prize before it expires! Act FAST! Send your bank details immediately!"
    },
    {
        "label": "Obvious not spam",
        "text": "Hi Rahul, just following up on our meeting yesterday. I've attached the updated project timeline. Let me know if the Thursday deadline works for your team. Best, Priya"
    },
    {
        "label": "Subtle spam (promotional)",
        "text": "Hey there! We noticed you signed up for our platform last month but haven't completed your profile. Complete it now and get 50% OFF our premium plan! Limited time offer. Unsubscribe link at bottom."
    },
    {
        "label": "Ambiguous (legit but urgent)",
        "text": "URGENT: Your account password will expire in 24 hours. Please log in to your company portal at portal.company.com and update your password. Contact IT support if you need help. - IT Department"
    },
]

print("=" * 80)
print("EMAIL SPAM CLASSIFICATION USING LOGPROBS")
print("=" * 80)

for email in emails:
    result = classify_email(email["text"])

    confidence = max(result["p_yes"], result["p_no"])
    is_spam = result["p_yes"] > result["p_no"]
    verdict = "SPAM" if is_spam else "NOT SPAM"

    confidence_label = (
        "high" if confidence > 0.90 else
        "medium" if confidence > 0.70 else
        "low (ambiguous)"
    )

    print(f"\n--- {email['label']} ---")
    print(f"Email: {email['text'][:80]}...")
    print(f"  P(Yes/Spam):     {result['p_yes']:.1%}")
    print(f"  P(No/Not Spam):  {result['p_no']:.1%}")
    print(f"  Verdict:         {verdict} (confidence: {confidence_label})")

    # Show all alternatives the model considered
    print(f"  All top tokens:  ", end="")
    for tok, prob in result["all_alternatives"]:
        print(f"{repr(tok.strip())}={prob:.1%}  ", end="")
    print()


print("\n" + "=" * 80)
print("WHY THIS MATTERS")
print("=" * 80)
print("""
Traditional approach:
  1. Collect labeled dataset (thousands of examples)
  2. Train a classifier (sklearn, pytorch)
  3. Deploy, monitor, retrain

Logprobs approach:
  1. Write a prompt
  2. Read logprobs
  3. Done

You get a calibrated probability score WITHOUT fine-tuning,
WITHOUT a labeled dataset, WITHOUT training infrastructure.

For an agent: any binary decision (route to agent A vs B,
escalate vs handle, approve vs reject) can be made this way.
The logprob IS the confidence. Use it.
""")
