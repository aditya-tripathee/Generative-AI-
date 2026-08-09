from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from Basic folder
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)



from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    # temperature controls the randomness and creativity of the model's output:
    # - Low (e.g., 0.0 - 0.3): Deterministic, focused, accurate. Best for factual QA, coding, math.
    # - High (e.g., 0.7 - 1.0+): Creative, diverse, random. Best for brainstorming, storytelling, creative writing.
    temperature=0.8,
    # max_output_tokens limits the maximum length of the generated response:
    # - 1 token is roughly 4 characters or ~0.75 words in English.
    # - Prevents overly long responses, controls API token costs, and caps output length.
    # - If the output reaches this limit, the response will stop (truncate).
    max_output_tokens=500
)

response = model.invoke("Who is Birendra Shah?")
print(response.content)

