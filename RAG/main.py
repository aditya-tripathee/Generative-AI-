from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI

load_dotenv()

chat = ChatMistralAI(model="mistral-small-latest")

res = chat.invoke("Hello, how are you?")
print(res)

