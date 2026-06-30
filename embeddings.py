import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

def get_bge_embeddings():
    """Returns the configured BGE embedding model."""
    # Using the small version for efficiency, but it still packs a punch!
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    
    # BGE models require normalized embeddings for best retrieval performance
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def save_vectorstore(documents):
    """Embeds documents and saves them to the local FAISS index."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    embeddings = get_bge_embeddings()
    print("Generating BGE embeddings and building FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved successfully to {VECTORSTORE_DIR}/")
    return vectorstore

def load_vectorstore():
    """Loads the local FAISS index for retrieval."""
    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        print("No vector store found.")
        return None
        
    embeddings = get_bge_embeddings()
    print("Loading existing FAISS vector store...")
    # Note: allow_dangerous_deserialization is required for local pickle files in recent LangChain updates
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
