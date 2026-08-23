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

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv())

# ============================================================
# MODEL
# ============================================================

model = ChatMistralAI(model="open-mistral-7b")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Funny AI Agent",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# CUSTOM CSS - UI ONLY
# ============================================================

st.markdown(
    """
    <style>
    /* Header */
    .chat-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        color: #ffffff !important;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.25);
    }

    .chat-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff !important;
    }

    .chat-header p {
        margin: 6px 0 0;
        opacity: 0.9;
        font-size: 15px;
        color: #f3f4f6 !important;
    }

    /* Sidebar info card */
    .bot-info {
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.2);
        padding: 14px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .bot-info-title {
        font-weight: 600;
        margin-bottom: 4px;
        font-size: 15px;
    }

    .bot-info-text {
        font-size: 13px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="chat-header">
        <h1>🤖 Funny AI Agent</h1>
        <p>Chat with your Mistral AI assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny ai agent.")
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 AI Agent")

    st.markdown(
        """
        <div class="bot-info">
            <div class="bot-info-title">Funny AI Agent</div>
            <div class="bot-info-text">Powered by Mistral AI (open-mistral-7b)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Model")
    st.code("open-mistral-7b")

    st.caption("Role")
    st.write("Funny AI Agent")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

# If no user messages yet, display a initial greeting
user_messages = [m for m in st.session_state.messages if isinstance(m, (HumanMessage, AIMessage))]
if not user_messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.write("Hey there! 👋 I am your Funny AI Agent. Ask me anything!")

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Type your message..."
)


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

if prompt:

    # Same functionality as:
    # messages.append(HumanMessage(content=prompt))

    human_message = HumanMessage(content=prompt)

    st.session_state.messages.append(human_message)

    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Invoke model using full conversation history
    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("Thinking..."):

            response = model.invoke(
                st.session_state.messages
            )

        # Same functionality as:
        # messages.append(AIMessage(content=response.content))

        ai_message = AIMessage(
            content=response.content
        )

        st.session_state.messages.append(
            ai_message
        )

        st.write(response.content)