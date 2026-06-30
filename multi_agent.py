import argparse
import json
import os
from langchain_core.prompts import PromptTemplate
from agent import build_agent, create_demo_database
from rag_agent import build_rag_agent
from compressor import get_llm

def create_router():
    """Builds a simple LLM router to classify user intent as structured (SQL) or unstructured (RAG)."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an intelligent router for a company assistant. "
        "Your job is to classify the user's question into one of two categories:\n"
        "1. 'sql' - The question requires querying a relational database (e.g. sales, revenue, users, customers, products, orders).\n"
        "2. 'rag' - The question requires looking up unstructured documents (e.g. policies, office locations, rules, guidelines, text).\n"
        "Output ONLY a valid JSON object with a single key 'route' and value either 'sql' or 'rag'. Do not output any other text or markdown.\n\n"
        "Question: {question}"
    )
    # This chain will output a JSON string
    return prompt | llm

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Master Router")
    parser.add_argument("--db", default="data/demo.sqlite", help="SQLite database path")
    parser.add_argument("--question", help="Question to ask (omit for interactive mode)")
    args = parser.parse_args()
    
    print("Loading Multi-Agent System...")
    
    # Ensure the data directory and dummy DB exist for the SQL agent
    if args.db == "data/demo.sqlite" and not os.path.exists("data/demo.sqlite"):
        os.makedirs("data", exist_ok=True)
        create_demo_database("data/demo.sqlite")
        
    print("  -> Initializing SQL Agent (Structured Data)...")
    try:
        sql_agent, _ = build_agent(args.db)
    except Exception as e:
        print(f"Error loading SQL Agent: {e}")
        return
        
    print("  -> Initializing RAG Agent (Unstructured Documents)...")
    try:
        rag_agent = build_rag_agent()
    except Exception as e:
        print(f"Error loading RAG Agent: {e}")
        return
        
    print("  -> Initializing Master Router...")
    router = create_router()
    
    print("\n[OK] Multi-Agent System Ready!")
    
    def process_query(q):
        # 1. Ask the LLM where this question should go
        try:
            route_res = router.invoke({"question": q}).content
            # Clean up potential markdown formatting from the LLM output
            route_res = route_res.replace('```json', '').replace('```', '').strip()
            route = json.loads(route_res).get("route", "rag")
        except Exception:
            # Fallback if the LLM hallucinates raw text like 'SQL' instead of JSON
            route = "rag"
            
        route = str(route).lower().strip()
        
        print(f"  [Router Decision: Sending query to the {route.upper()} Agent]")
        
        # 2. Hand the query to the chosen expert agent with fallback support
        if route == "sql":
            try:
                res = sql_agent.invoke({"input": q})
                return res["output"]
            except Exception as e:
                print(f"  [SQL Agent failed: {e}. Falling back to RAG Agent...]")
                try:
                    res = rag_agent.invoke({"input": q})
                    return res["answer"]
                except Exception as e2:
                    return f"Both agents failed. SQL Error: {e}, RAG Error: {e2}"
        else:
            try:
                res = rag_agent.invoke({"input": q})
                return res["answer"]
            except Exception as e:
                print(f"  [RAG Agent failed: {e}. Falling back to SQL Agent...]")
                try:
                    res = sql_agent.invoke({"input": q})
                    return res["output"]
                except Exception as e2:
                    return f"Both agents failed. RAG Error: {e}, SQL Error: {e2}"
            
    if args.question:
        print(f"\nUser: {args.question}")
        answer = process_query(args.question)
        print(f"\nAgent: {answer}")
    else:
        print("\nMulti-Agent Chat ready. Ask about sales, users, or company policies! Type 'quit' to exit.")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in ["quit", "exit", "q"]:
                break
            if not q:
                continue
                
            answer = process_query(q)
            print(f"\nAgent: {answer}")

if __name__ == "__main__":
    main()
