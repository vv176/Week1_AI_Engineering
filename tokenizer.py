import tiktoken

text = "Database normalization is asked in faang interviews"
enc = tiktoken.encoding_for_model("gpt-4o")  # pick the model
tokens = enc.encode(text)

print("Text:", text)
print("Tokens:", [enc.decode([t]) for t in tokens])
print("Token IDs:", tokens)


