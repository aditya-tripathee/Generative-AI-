from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print

# creating a tool
@tool
def get_text_length(text:str) -> int:
    """Get the length of a text"""
    return len(text)


llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7,
)

# tool binding 

llm_with_tools = llm.bind_tools([get_text_length])



print("\n" + "="*50 + "\n")

print(llm_with_tools.invoke("Hi, how are you?"))



llm.invoke("Hello, how are you?")


