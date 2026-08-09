from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from Basic folder
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)



from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

response = model.invoke("Who is Balen Shah?")
print(response.content)