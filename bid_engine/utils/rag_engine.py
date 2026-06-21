# utils/rag_engine.py
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.data_loader import load_capability_library

VECTOR_STORE_PATH = "vector_store/faiss_index.pkl"

# We use a small, fast model — good enough for a hackathon
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store():
    """Build FAISS index from the Capability Library. Call once at startup."""
    df = load_capability_library()
    texts = df['embed_text'].tolist()

    print("⚙️  Generating embeddings for Capability Library...")
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index (flat L2 = exact nearest neighbour, fine for 50 records)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + metadata side by side
    os.makedirs("vector_store", exist_ok=True)
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump({"index": index, "texts": texts, "df": df}, f)

    print(f"✅ Vector store built! {len(texts)} capabilities indexed.")
    return index, texts, df

def load_vector_store():
    """Load existing vector store from disk."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return build_vector_store()
    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["texts"], data["df"]

def search_capabilities(query: str, top_k: int = 5):
    """
    Given a requirement string from an RFP, find the top_k most
    relevant capabilities from our library.
    Returns a list of dicts with cap details + similarity score.
    """
    index, texts, df = load_vector_store()

    query_vec = EMBED_MODEL.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        results.append({
            "cap_id": row["Cap ID"],
            "domain": row["Domain"],
            "summary": row["Project Summary"],
            "certification": row["Certification"],
            "client_type": row["Client Type"],
            "contract_value": row["Contract Value"],
            "score": float(1 / (1 + dist))  # convert L2 distance → similarity score 0-1
        })
    return results