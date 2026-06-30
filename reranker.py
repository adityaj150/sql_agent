from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from retriever import get_hybrid_retriever

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading Cross-Encoder model into memory...")
        _cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def get_reranking_retriever():
    """
    Wraps the Hybrid Retriever with a CrossEncoder to re-score and re-rank
    the retrieved documents based on exact semantic relevance to the query.
    """
    print("Initializing Cross-Encoder Re-ranker...")
    
    # Use the cached model to avoid reloading on every query
    model = get_cross_encoder()
    
    # We want to keep only the absolute best 3 documents after re-ranking
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    hybrid_retriever = get_hybrid_retriever()
    
    # The contextual compression retriever automatically routes the initial query 
    # to our Hybrid Retriever, gets the top 5 results, and then re-ranks them.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )
    
    return compression_retriever
