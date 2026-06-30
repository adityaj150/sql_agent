import os
import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from embeddings import load_vectorstore

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
BM25_FILE = os.path.join(VECTORSTORE_DIR, "bm25_retriever.pkl")

def save_bm25_retriever(documents):
    """Builds and saves the BM25 keyword retriever locally."""
    print("Generating BM25 index for sparse retrieval...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    bm25_retriever = BM25Retriever.from_documents(documents)
    # We will retrieve the top 5 documents based on keyword matches
    bm25_retriever.k = 5 
    
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"BM25 Retriever saved successfully to {BM25_FILE}")
    return bm25_retriever

def load_bm25_retriever():
    """Loads the pre-built BM25 index."""
    if not os.path.exists(BM25_FILE):
        print("No BM25 index found.")
        return None
    with open(BM25_FILE, "rb") as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever

def get_hybrid_retriever():
    """Combines BM25 and Vector Search into a hybrid retriever."""
    print("Initializing Hybrid Retriever (BM25 + FAISS)...")
    
    vectorstore = load_vectorstore()
    bm25_retriever = load_bm25_retriever()
    
    if vectorstore is None or bm25_retriever is None:
        raise ValueError("Indices missing! Please run ingest.py to build the database.")
        
    # We will also retrieve the top 5 semantically similar documents
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # The ensemble retriever merges results, re-ranking them mathematically.
    # 30% weight to exact keyword matches, 70% weight to semantic meaning
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7] 
    )
    
    return ensemble_retriever
