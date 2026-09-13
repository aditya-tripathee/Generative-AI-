
from langchain.tools import tool

@tool("greeting")
def get_greeting(name:str) -> str:
    """
    Generate a greeting message for a user
    """
    return f"Hello {name} Welcome to the world of LangChain Tools."

print(get_greeting.invoke({"name":"Aditya"}))

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args_schema)


