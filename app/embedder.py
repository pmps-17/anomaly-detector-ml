from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate embeddings for given text."""
    return model.encode([text])[0]
