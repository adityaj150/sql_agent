import streamlit as st
from multi_agent import init_system, ask_multi_agent

st.set_page_config(
    page_title="Multi-Agent AI Assistant",
    page_icon="🤖",
    layout="centered"
)

import os
from ingest import run_ingestion

# Session state for tracking the database path
if "db_path" not in st.session_state:
    st.session_state.db_path = "data/demo.sqlite"

# Use st.cache_resource so we only load the heavy ML models and DB connections once!
@st.cache_resource(show_spinner="Booting up AI Agents...")
def load_agents(db_path):
    return init_system(db_path)

try:
    sql_agent, rag_agent, router = load_agents(st.session_state.db_path)
except Exception as e:
    st.error(f"Failed to initialize agents: {e}")
    st.stop()

st.title("Multi-Agent AI Assistant 🤖")
st.markdown("Ask a question about our structured database (SQL) or unstructured documents (RAG). The Master Router will automatically decide which expert agent to use!")

with st.sidebar:
    st.header("Upload Data")
    st.caption("Upload your own files to replace the dummy data!")
    
    # Document uploader
    doc_file = st.file_uploader("Upload Company Policy (.md, .txt, .pdf)", type=["md", "txt", "pdf"])
    if doc_file is not None:
        if st.button("Process Document"):
            os.makedirs("documents", exist_ok=True)
            file_path = os.path.join("documents", doc_file.name)
            with open(file_path, "wb") as f:
                f.write(doc_file.getbuffer())
            with st.spinner("Indexing document into Vector Database..."):
                run_ingestion()
                st.cache_resource.clear()
            st.success(f"Successfully indexed {doc_file.name}!")
            st.rerun()
            
    st.divider()
    
    # SQL uploader
    db_file = st.file_uploader("Upload SQLite Database (.sqlite, .db)", type=["sqlite", "db"])
    if db_file is not None:
        if st.button("Connect Database"):
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", db_file.name)
            with open(file_path, "wb") as f:
                f.write(db_file.getbuffer())
            st.session_state.db_path = file_path
            st.cache_resource.clear()
            st.success(f"Successfully connected to {db_file.name}!")
            st.rerun()

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "route" in message:
            st.caption(f"🧠 *Routed via {message['route'].upper()} Agent*")

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Agent is thinking..."):
        # The magic happens here!
        route, response = ask_multi_agent(prompt, sql_agent, rag_agent, router)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"🧠 *Routed via {route.upper()} Agent*")
        
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response, 
        "route": route
    })
