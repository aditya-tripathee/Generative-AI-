from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI

load_dotenv()

chat = ChatMistralAI(model_name="mistral-7b-instruct")

res = chat.invoke("Hello, how are you?")
print(res)

