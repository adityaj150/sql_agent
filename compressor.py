from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from reranker import get_reranking_retriever
from dotenv import load_dotenv

load_dotenv()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        # Same configuration as the SQL agent
        _llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    return _llm

def get_compressed_retriever():
    """
    Wraps the Cross-Encoder Reranker with an LLM-based context compressor.
    This extracts ONLY the exact sentences relevant to the query from the top documents,
    stripping away noise to save tokens and improve final generation accuracy.
    """
    print("Initializing LLM Context Compressor...")
    
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    
    # We build upon the reranker, so we only compress the absolute best 3 documents
    base_retriever = get_reranking_retriever()
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
