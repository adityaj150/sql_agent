import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chunking import chunk_documents
from langchain_core.documents import Document

def load_documents(docs_dir: str = "documents"):
    """Loads all PDF, TXT, and MD files from the given directory."""
    documents = []
    
    if not os.path.exists(docs_dir):
        print(f"Directory '{docs_dir}' does not exist. Please add documents.")
        return documents
        
    for filepath in glob.glob(os.path.join(docs_dir, "*")):
        ext = filepath.lower().split('.')[-1]
        
        if ext == 'pdf':
            print(f"Loading PDF: {filepath}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            
        elif ext in ['txt', 'md']:
            print(f"Loading Text/Markdown: {filepath}")
            # Try utf-8 first, fallback if necessary
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
        else:
            print(f"Skipping unsupported file type: {filepath}")
            
    return documents

if __name__ == "__main__":
    print("Starting document ingestion pipeline...")
    docs = load_documents()
    
    if docs:
        print(f"Loaded {len(docs)} document pages/files.")
        chunks = chunk_documents(docs)
        print(f"Chunked into {len(chunks)} context-aware segments.")
        
        # Save to local FAISS vector store
        from embeddings import save_vectorstore
        save_vectorstore(chunks)
        
        # Save to local BM25 store
        from retriever import save_bm25_retriever
        save_bm25_retriever(chunks)
        
        print("Ingestion complete!")
    else:
        print(f"No documents found in the 'documents/' directory.")
