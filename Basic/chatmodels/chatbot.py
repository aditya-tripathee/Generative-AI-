import sys
from dotenv import load_dotenv, find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(find_dotenv())

model = ChatMistralAI(model="open-mistral-7b")

"""
===================================================================
DEFINITION OF MESSAGE TYPES IN GEN AI / LANGCHAIN:
===================================================================
1. SystemMessage:
   - High-priority directive sent to the LLM at the start of a chat.
   - Defines the bot's persona, rules, role, constraints, and tone.
   - Example: SystemMessage(content="You are a helpful Python tutor.")

2. HumanMessage:
   - Represents the input/prompt sent by the human user.
   - Example: HumanMessage(content="What is a loop?")

3. AIMessage:
   - Represents the response returned by the AI model.
   - Example: AIMessage(content="A loop repeats a block of code...")
===================================================================
"""

# SystemMessage: Sets the initial behavior and guardrails for the chatbot
messages = [
    SystemMessage(content="You are a funny ai agent.")
]

print("------------ welcome type 0 to exit the chat ------------")
while True:
  
    prompt = input("You : ")
    if prompt == '0':
        print("Bot: Exiting the chat")
        break

    # HumanMessage: Wraps user input into a HumanMessage object
    messages.append(HumanMessage(content=prompt))

    # Invoke model with full message history (SystemMessage + HumanMessages + AIMessages)
    response = model.invoke(messages)
    
    # AIMessage: Append the AI response object back into history
    messages.append(AIMessage(content=response.content))

    print("Bot: ", response.content)


print(messages)