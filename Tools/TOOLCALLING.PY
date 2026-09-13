from dotenv import load_dotenv
load_dotenv()

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


# ── 1. Define a Tool ─────────────────────────────────────────────────────────
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in the given text."""
    return len(text)


# ── 2. Create LLM and Bind Tool ───────────────────────────────────────────────
llm = ChatMistralAI(model="open-mistral-7b", temperature=0)
llm_with_tools = llm.bind_tools([get_text_length])

print("\n[bold cyan]Tool Info:[/bold cyan]")
print(f"  Name       : {get_text_length.name}")
print(f"  Description: {get_text_length.description}")
print(f"  Args Schema: {get_text_length.args}")

print("\n" + "="*60)

# ── 3. Invoke LLM with a query that TRIGGERS the tool ─────────────────────────
query = "What is the length of the text 'LangChain Tool Calling'?"
print(f"\n[bold yellow]User Query:[/bold yellow] {query}\n")

messages = [HumanMessage(content=query)]
ai_response = llm_with_tools.invoke(messages)

print("[bold green]AI Response (Step 1 — tool call decision):[/bold green]")
print(ai_response)

# ── 4. Execute the Tool if LLM requested it ───────────────────────────────────
if ai_response.tool_calls:
    print(f"\n[bold magenta]Tool Calls Requested:[/bold magenta] {ai_response.tool_calls}")

    # Execute each requested tool
    tool_results = []
    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id   = tool_call["id"]

        print(f"\n  Calling tool '{tool_name}' with args: {tool_args}")
        result = get_text_length.invoke(tool_args)
        print(f"  Tool result: {result}")

        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    # ── 5. Send Tool Results back to LLM for Final Answer ─────────────────────
    messages = messages + [ai_response] + tool_results
    final_response = llm_with_tools.invoke(messages)

    print("\n[bold green]Final LLM Answer (Step 2 — after tool result):[/bold green]")
    print(final_response.content)

else:
    # LLM answered directly without calling any tool
    print("\n[bold green]Direct Answer (no tool called):[/bold green]")
    print(ai_response.content)
