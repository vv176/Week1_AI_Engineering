from openai import OpenAI
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "Thank you for your help."
enc = tiktoken.encoding_for_model("gpt-4o")  # pick the model
tokens = enc.encode(text)

token_embeddings = []
for tok in tokens:
    tok_str = enc.decode([tok])
    emb = client.embeddings.create(model="text-embedding-3-small", input=tok_str)  #large
    token_embeddings.append(emb.data[0].embedding)

# Print size of embeddings
print("Size of embeddings:", len(token_embeddings[0]))

# Print first ten dimensions of all embeddings
print("First ten dimensions of all embeddings:")
for i, embedding in enumerate(token_embeddings):
    print(f"Token {i}: {embedding[:10]}")

#print("Embedding of 'ball':", token_embeddings[1][:10])  # first 10 dims
print("First 5 dims of Embedding of 'I am happy':", client.embeddings.create(model="text-embedding-3-small", input="I am happy").data[0].embedding[:5])

