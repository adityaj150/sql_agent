import streamlit as st
from multi_agent import init_system, ask_multi_agent

st.set_page_config(
    page_title="Multi-Agent AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Use st.cache_resource so we only load the heavy ML models and DB connections once!
@st.cache_resource(show_spinner="Booting up AI Agents...")
def load_agents():
    return init_system()

try:
    sql_agent, rag_agent, router = load_agents()
except Exception as e:
    st.error(f"Failed to initialize agents: {e}")
    st.stop()

st.title("Multi-Agent AI Assistant 🤖")
st.markdown("Ask a question about our structured database (SQL) or unstructured documents (RAG). The Master Router will automatically decide which expert agent to use!")

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
