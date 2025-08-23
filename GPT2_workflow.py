
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_ID = "gpt2"  # small, runs on CPU; you can use gpt2-medium/large on GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.to(device).eval() # prepares the GPT-2 model for inference by placing it on the right hardware and setting it to evaluation mode(and not training mode by diabling dropouts etc...) for consistent text generation

text = "I love Maths because"
enc = tokenizer(text, return_tensors="pt")
input_ids = enc["input_ids"].to(device)          # [1, seq_len]
attn_mask = enc["attention_mask"].to(device)

# 1) Tokens & IDs
tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
print("Text:", text)
print("Tokens:    ", tokens)
print("Token IDs: ", input_ids[0].tolist())

# 2) Token embedding matrix (learned)
#    Each row is the vector for one token ID (size = hidden_dim, 768 for gpt2)
wte = model.transformer.wte.weight               # [vocab_size, hidden_dim], uncontextualised and w/o positional embeddings

# Dump first 100 tokens and their embeddings to TSV file
print("Dumping first 100 tokens and embeddings to 'token_embeddings.tsv'...")
with open('token_embeddings.tsv', 'w') as f:
    # Write header
    f.write("Token_ID\tToken_Text\tEmbedding_Vector\n")
    
    # Loop through first 100 tokens
    for token_id in range(100):
        # Get the token text
        token_text = tokenizer.decode([token_id])
        # Get the embedding vector (convert to list for easier viewing)
        embedding_vector = wte[token_id].detach().cpu().numpy().tolist()
        # Write to TSV
        f.write(f"{token_id}\t{token_text}\t{embedding_vector}\n")

print("TSV file created successfully!")

tok_emb = wte[input_ids]                         # [1, seq_len, hidden_dim]
print("Token embedding shape:", tok_emb.shape)
print("\nFirst 10 dimensions of each token in 'May God bless':")
for i, token in enumerate(tokens):
    token_embedding = tok_emb[0, i]  # Get embedding for token at position i
    first_10_dims = token_embedding[:10].detach().cpu().numpy()
    print(f"Token {i}: {repr(token):<15} | First 10 dims: {[round(x, 4) for x in first_10_dims]}")


# 3) Positional embedding matrix (learned absolute positions 0..1023)
#    One vector per position index; added to token embeddings
wpe = model.transformer.wpe.weight               # [max_positions, hidden_dim]
seq_len = input_ids.size(1)
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
print("Position IDs:", position_ids)
pos_emb = wpe[position_ids]                      # [1, seq_len, hidden_dim]
print("Positional embedding shape:", pos_emb.shape)

# Peek at a few dims to show they vary by position
# Print first 10 dimensions of positional embeddings for each position
print("\nFirst 10 dimensions of positional embeddings for each position:")
for i in range(seq_len):
    pos_embedding = pos_emb[0, i]  # Get positional embedding for position i
    first_10_dims = pos_embedding[:10].detach().cpu().numpy()
    print(f"Position {i}: {repr(tokens[i]):<15} | First 10 dims: {[round(x, 4) for x in first_10_dims]}")

# 4) Input to the first block = token_emb + pos_emb (what GPT-2 actually uses)
inputs_embeds = tok_emb + pos_emb
print("Input (token+pos) shape:", inputs_embeds.shape)
for i in range(seq_len):
    input_embedding = inputs_embeds[0, i]  # Get combined embedding for position i
    first_10_dims = input_embedding[:10].detach().cpu().numpy()
    print(f"Position {i}: {repr(tokens[i]):<15} | First 10 dims: {[round(x, 4) for x in first_10_dims]}")
# 5) Contextual embeddings (hidden states at every layer)
with torch.no_grad():
    out = model(inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True)

hidden_states = out.hidden_states  # list length = n_layers + 1
print("Layer 0 (post-embed) :", hidden_states[0].shape)     # after tok+pos
print("Final layer (context):", hidden_states[-1].shape)     # contextual embeddings

print("\nFirst 10 dimensions of contextual embeddings from each hidden layer:")
for layer_idx, hidden_state in enumerate(hidden_states):
    layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx-1}" # The embedding layer does more than just addition: The embedding layer in GPT-2 includes: Token + Positional addition (tok_emb + pos_emb), Layer normalization, Dropout (though disabled in eval mode), Other embedding-specific processing
    print(f"\n{layer_name} (shape: {hidden_state.shape}):")
    
    for pos_idx in range(seq_len):
        token_name = tokens[pos_idx]
        contextual_embedding = hidden_state[0, pos_idx]  # Get embedding for position pos_idx
        first_10_dims = contextual_embedding[:10].detach().cpu().numpy()
        print(f"  Position {pos_idx}: {repr(token_name):<15} | First 10 dims: {[round(x, 4) for x in first_10_dims]}")



# 6) Next-token prediction (distribution over vocab)
logits_all = out.logits[0]  # [seq_len, vocab_size]
print("\nTop-5 next tokens per input position:")
for i in range(seq_len):
    probs_i  = torch.softmax(logits_all[i], dim=-1)
    topk_i   = torch.topk(probs_i, k=5)
    print(f"\nPosition {i} (after {repr(tokens[i])}):")
    for tid, p in zip(topk_i.indices.tolist(), topk_i.values.tolist()):
        print(f"  {repr(tokenizer.decode([tid]))}: {p:.4f}")

# 7) Ablation: run the model WITHOUT adding positional embeddings (to show necessity)
with torch.no_grad():
    out_no_pos = model(inputs_embeds=tok_emb,  # <-- no pos_emb added
                       attention_mask=attn_mask,
                       output_hidden_states=False,
                       return_dict=True)
probs_no_pos = torch.softmax(out_no_pos.logits[:, -1, :], dim=-1) # logits can be negative, so we need to softmax to get probabilities, softmax formula is e^x / sum(e^x)
topk2 = torch.topk(probs_no_pos, k=5)
print("\nTop-5 next tokens WITHOUT positional embeddings:")
for tid, p in zip(topk2.indices[0].tolist(), topk2.values[0].tolist()):
    print(f"{repr(tokenizer.decode([tid]))}: {p:.4f}")

# 8) Simple greedy generation loop (argmax)
max_new_tokens = 20
current_ids = input_ids.clone()
print("\nGreedy generation (argmax) up to", max_new_tokens, "tokens:")
for step in range(max_new_tokens):
    with torch.no_grad():
        step_out = model(input_ids=current_ids, return_dict=True)
    next_id = torch.argmax(step_out.logits[:, -1, :], dim=-1)  # [1]
    token_id = int(next_id.item())
    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
    print(f"Step {step+1}: {repr(token_text)}")

    # Append token to sequence
    current_ids = torch.cat([current_ids, next_id.unsqueeze(0)], dim=1)

    # Stop on EOS or sentence-ending punctuation
    if token_id == tokenizer.eos_token_id:
        print("Stopping: EOS token")
        break
    full_text_now = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    if full_text_now.rstrip().endswith((".", "!", "?")):
        print("Stopping: sentence end")
        break

final_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
print("\nFinal completion:")
print(final_text)
