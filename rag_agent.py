import argparse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from compressor import get_llm, get_compressed_retriever

def build_rag_agent():
    """Builds the final RAG chain connecting the LLM to our advanced retrieval pipeline."""
    llm = get_llm()
    
    # Get the top-level retriever which has FAISS -> BM25 -> CrossEncoder -> Context Compressor built in!
    retriever = get_compressed_retriever()
    
    system_prompt = (
        "You are a highly intelligent and helpful company assistant. "
        "Use the following pieces of retrieved context to answer the user's question accurately. "
        "If the answer is not contained in the context, just say that you don't know. "
        "Do not make up information. Keep the answer concise and direct.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # The 'stuff' chain takes the documents and formats them into the {context} variable of the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # The retrieval chain automatically orchestrates everything: User Query -> Retriever -> Context -> LLM -> Answer
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Chatbot")
    parser.add_argument("--question", help="Natural language question (omit for interactive mode)")
    args = parser.parse_args()

    print("Loading RAG Agent pipeline...")
    try:
        rag_chain = build_rag_agent()
    except Exception as e:
        print(f"Error loading RAG Agent: {e}")
        print("Did you remember to run 'python ingest.py' first?")
        return

    print("\n[OK] RAG Agent ready!")

    if args.question:
        print(f"Question: {args.question}")
        response = rag_chain.invoke({"input": args.question})
        print(f"\nAnswer: {response['answer']}")
    else:
        print("RAG Agent ready. Ask questions about your documents! Type 'quit' to exit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
                
            response = rag_chain.invoke({"input": question})
            print(f"\nAgent: {response['answer']}\n")

if __name__ == "__main__":
    main()
