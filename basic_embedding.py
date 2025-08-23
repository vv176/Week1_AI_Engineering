from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
sentences = [
  "Dog",
  "Puppy",
  "Cat",
  "Kitten",
  "Cricket",
  "Wicket-keeper",
]
E = model.encode(sentences, normalize_embeddings=True)  # shape: (n, 384)

# 2D for plotting (PCA)
xy = PCA(n_components=2).fit_transform(E)
for s, (x, y) in zip(sentences, xy):
    print(f"{x:+.2f}\t{y:+.2f}\t{s}")

# Plot the embeddings
plt.figure(figsize=(10, 8))
plt.scatter(xy[:, 0], xy[:, 1], c='blue', s=100, alpha=0.7)

# Add labels for each point
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (xy[i, 0], xy[i, 1]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Word Embeddings Visualization (PCA)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cosine similarity example
def cos(a,b): return float(np.dot(a,b))
print(cos(E[0], E[1]))
print(cos(E[0], E[2]))
print(cos(E[0], E[4]))



