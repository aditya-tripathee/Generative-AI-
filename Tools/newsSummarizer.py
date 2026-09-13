from dotenv import load_dotenv
load_dotenv()

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent


# 1. Tools
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]

# 2. LLM
llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7,
)

# 3. Create agent (LangChain 1.x — LangGraph-based, no AgentExecutor needed)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that searches and summarizes the latest news."
)

# 4. Run the agent — input must be {"messages": [...]}
inputs = {"messages": [{"role": "user", "content": "Who is the Baji rao peshwa 1?"}]}

print("=== Agent Running ===\n")
for chunk in agent.stream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        messages = values.get("messages", [])
        for msg in messages:
            content = getattr(msg, "content", "")
            if content:
                print(f"[{node}]: {content}\n")

print("=== Done ===")
