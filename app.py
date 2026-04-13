import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent import build_agent

st.set_page_config(page_title="AutoStream Sales Agent", page_icon="🤖", layout="centered")

st.title("🤖 AutoStream Sales Agent")
st.markdown("Your personal conversational agent for video editing SaaS pricing & leads.")

# Groq API Key UI
api_key = st.text_input("Enter your Groq API Key to begin:", type="password")

if not api_key:
    st.info("Don't have a key? Get one for free at [console.groq.com](https://console.groq.com/keys)")
    st.stop()
else:
    # Initialize agent for the first time
    if "agent_app" not in st.session_state:
        try:
            with st.spinner("Initializing Groq models and fetching RAG context..."):
                st.session_state.agent_app = build_agent(api_key=api_key, model_name="llama-3.1-8b-instant")
        except Exception as e:
            st.error(f"Failed to initialize Groq. Please check your API key. Error: {e}")
            st.stop()

# Initialize session state for memory Checkpointer
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        response_text = ""
        with st.spinner("Processing via Groq LLM..."):
            try:
                for state_update in st.session_state.agent_app.stream(inputs, config):
                    for node_name, values in state_update.items():
                        if node_name in ["greeting", "inquiry", "lead"]:
                            ai_msg = values["messages"][-1]
                            response_text = ai_msg.content
            except Exception as e:
                response_text = f"An error occurred during process: {e}"
        
        message_placeholder.markdown(response_text)
        st.session_state.messages.append(AIMessage(content=response_text))
