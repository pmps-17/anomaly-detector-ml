import faiss
import numpy as np
import pickle

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

def build_faiss_index(documents):
    """Embeds documents and saves FAISS index and metadata."""
    embeddings = np.array([doc["embedding"] for doc in documents]).astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    # Save metadata (original text)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(documents, f)

def load_index():
    """Loads FAISS index and metadata."""
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def query_index(query_embedding, top_k=3):
    """Queries the FAISS index for similar documents."""
    index, metadata = load_index()

    q_vec = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, top_k)

    return [metadata[i]["text"] for i in indices[0]]
