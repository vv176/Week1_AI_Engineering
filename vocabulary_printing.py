import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

# 1) Vocabulary sizes
print("Total vocab size:", enc.n_vocab)
print("Mergeable tokens:", len(enc._mergeable_ranks))
print("Special tokens:", len(enc._special_tokens))

# 2) Dump all token IDs -> strings (writes to a TSV to avoid spamming stdout)
id_to_text = {}

# Normal (mergeable) tokens
for b, tok_id in enc._mergeable_ranks.items():
    id_to_text[tok_id] = b.decode("utf-8", errors="replace")  # b is of type 'bytes'

# Special tokens
for s, tok_id in enc._special_tokens.items():
    id_to_text[tok_id] = f"<SPECIAL {s}>"

with open("vocab_dump.tsv", "w", encoding="utf-8") as f:
    f.write("token_id\ttext\n")
    for tok_id in range(5000):
        text = id_to_text.get(tok_id)
        if text is None:
            try:
                text = enc.decode_single_token_bytes(tok_id).decode("utf-8", errors="replace")
            except Exception:
                text = "<UNKNOWN>"
        f.write(f"{tok_id}\t{text}\n")

print("Wrote all tokens to vocab_dump.tsv")
