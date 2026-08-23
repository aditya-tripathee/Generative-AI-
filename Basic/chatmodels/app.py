import os
import sys

# Disable Streamlit watcher globally to prevent torchvision/transformers warnings
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st

# Automatically launch `streamlit run` if file is executed directly with `python script.py`
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__), "--server.fileWatcherType=none"]
        sys.exit(stcli.main())
except Exception:
    pass

from dotenv import load_dotenv, find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(find_dotenv())

# Configure Streamlit page layout
st.set_page_config(page_title="Funny AI Agent Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Funny AI Agent Chatbot")

# Initialize ChatMistralAI model
@st.cache_resource
def get_model():
    return ChatMistralAI(model="open-mistral-7b")

model = get_model()

# Initialize message history with SystemMessage in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny ai agent.")
    ]

# Display existing chat messages (excluding SystemMessage)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append HumanMessage object to message history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Invoke model with full message history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.messages)
            st.markdown(response.content)

    # Append AIMessage object to message history
    st.session_state.messages.append(AIMessage(content=response.content))
