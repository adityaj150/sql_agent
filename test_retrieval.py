from compressor import get_compressed_retriever

def test_search():
    print("\n--- Testing Compressed Retriever ---")
    retriever = get_compressed_retriever()
    
    query = "Where is the European Branch?"
    print(f"\nQuery: '{query}'\n")
    
    results = retriever.invoke(query)
    
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 30)

if __name__ == "__main__":
    test_search()
