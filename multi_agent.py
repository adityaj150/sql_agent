import argparse
import json
import os
from langchain_core.prompts import PromptTemplate
from agent import build_agent, create_demo_database
from rag_agent import build_rag_agent
from compressor import get_llm

def create_router():
    """Builds a simple LLM router to classify user intent as structured (SQL), unstructured (RAG), both, or general."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an intelligent router for a company assistant. "
        "Your job is to classify the user's question into one of four categories:\n"
        "1. 'sql' - The question requires querying a relational database (e.g. sales, revenue, users, customers, products, orders).\n"
        "2. 'rag' - The question requires looking up unstructured documents (e.g. policies, office locations, rules, guidelines, text).\n"
        "3. 'both' - The question bridges both domains (e.g. 'Which country generated the most revenue and what is our refund policy there?').\n"
        "4. 'general' - A general conversational question or greeting (e.g. 'Hi', 'Who are you?').\n"
        "Output ONLY a valid JSON object with a single key 'route' and value 'sql', 'rag', 'both', or 'general'. Do not output any other text or markdown.\n\n"
        "Question: {question}"
    )
    # This chain will output a JSON string
    return prompt | llm

def init_system(db_path="data/demo.sqlite"):
    if db_path == "data/demo.sqlite" and not os.path.exists("data/demo.sqlite"):
        os.makedirs("data", exist_ok=True)
        create_demo_database("data/demo.sqlite")
    sql_agent, _ = build_agent(db_path)
    rag_agent = build_rag_agent()
    router = create_router()
    return sql_agent, rag_agent, router

def ask_multi_agent(q, sql_agent, rag_agent, router, chat_history_str=""):
    # Inject chat history into the Router's classification prompt so it understands follow-ups
    router_q = q
    if chat_history_str:
        router_q = f"Previous Chat History:\n{chat_history_str}\n\nLatest Question: {q}"
        
    try:
        route_res = router.invoke({"question": router_q}).content
        route_res = route_res.replace('```json', '').replace('```', '').strip()
        route = json.loads(route_res).get("route", "rag")
    except Exception:
        route = "rag"
        
    route = str(route).lower().strip()
    
    # Inject chat history into the Agents' prompts so they can answer follow-ups correctly
    agent_q = q
    if chat_history_str:
        agent_q = f"Previous Chat History:\n{chat_history_str}\n\nLatest Question: {q}"
    
    if route == "general":
        try:
            llm = get_llm()
            res = llm.invoke(agent_q)
            return route, res.content
        except Exception as e:
            return "error", f"General LLM failed: {e}"
            
    elif route == "both":
        sql_ans = "Failed to fetch SQL data."
        try:
            sql_res = sql_agent.invoke({"input": agent_q})
            sql_ans = sql_res["output"]
        except Exception as e:
            sql_ans = f"SQL Agent error: {e}"
            
        # Create a targeted query for the RAG agent using the SQL context!
        llm = get_llm()
        rag_query_prompt = PromptTemplate.from_template(
            "Original Question: {question}\n"
            "Database Answer: {sql_answer}\n"
            "Based on the database answer, formulate a targeted search query to look up the relevant company policies needed to fully answer the original question (e.g. refund policies for specific products). Output ONLY the search query."
        )
        rag_q = (rag_query_prompt | llm).invoke({"question": agent_q, "sql_answer": sql_ans}).content.strip()
        
        rag_ans = "Failed to fetch RAG data."
        try:
            rag_res = rag_agent.invoke({"input": rag_q})
            rag_ans = rag_res["answer"]
        except Exception as e:
            rag_ans = f"RAG Agent error: {e}"
            
        combiner_prompt = PromptTemplate.from_template(
            "You are an intelligent synthesis assistant. You have received answers from two expert systems regarding the user's question.\n\n"
            "User's Question: {question}\n\n"
            "SQL Agent (Database) Answer:\n{sql_answer}\n\n"
            "RAG Agent (Documents) Answer:\n{rag_answer}\n\n"
            "Please synthesize these into a single, cohesive, and helpful response."
        )
        chain = combiner_prompt | llm
        try:
            final_ans = chain.invoke({"question": agent_q, "sql_answer": sql_ans, "rag_answer": rag_ans}).content
            return route, final_ans
        except Exception as e:
            return "error", f"Combiner LLM failed: {e}"

    elif route == "sql":
        try:
            res = sql_agent.invoke({"input": agent_q})
            return route, res["output"]
        except Exception as e:
            try:
                res = rag_agent.invoke({"input": agent_q})
                return "rag (fallback from sql)", res["answer"]
            except Exception as e2:
                return "error", f"Both agents failed. SQL Error: {e}, RAG Error: {e2}"
    else:
        try:
            res = rag_agent.invoke({"input": agent_q})
            return route, res["answer"]
        except Exception as e:
            try:
                res = sql_agent.invoke({"input": agent_q})
                return "sql (fallback from rag)", res["output"]
            except Exception as e2:
                return "error", f"Both agents failed. RAG Error: {e}, SQL Error: {e2}"

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Master Router")
    parser.add_argument("--db", default="data/demo.sqlite", help="SQLite database path")
    parser.add_argument("--question", help="Question to ask (omit for interactive mode)")
    args = parser.parse_args()
    
    print("Loading Multi-Agent System...")
    try:
        sql_agent, rag_agent, router = init_system(args.db)
    except Exception as e:
        print(f"Error initializing system: {e}")
        return
    
    print("\n[OK] Multi-Agent System Ready!")
    
    def process_query(q):
        route, ans = ask_multi_agent(q, sql_agent, rag_agent, router)
        print(f"  [Router Decision: Sending query to the {route.upper()} Agent]")
        return ans
            
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
