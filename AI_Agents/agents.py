from dotenv import load_dotenv
load_dotenv()

import sys, io, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent


# ── 1. Fake/Mock Weather Tool (no real API needed) ────────────────────────────
FAKE_WEATHER_DB = {
    "london":    {"temp": "15°C", "condition": "Cloudy", "humidity": "80%"},
    "new york":  {"temp": "22°C", "condition": "Sunny",  "humidity": "55%"},
    "tokyo":     {"temp": "28°C", "condition": "Humid",  "humidity": "75%"},
    "paris":     {"temp": "18°C", "condition": "Rainy",  "humidity": "85%"},
    "mumbai":    {"temp": "32°C", "condition": "Hot",    "humidity": "90%"},
    "kathmandu": {"temp": "20°C", "condition": "Clear",  "humidity": "60%"},
}

@tool
def get_current_weather(city_name: str) -> str:
    """
    Get the current weather for a specific city.

    Args:
        city_name (str): The name of the city to get weather for.

    Returns:
        str: A weather report string for the given city.
    """
    key = city_name.lower().strip()
    if key in FAKE_WEATHER_DB:
        w = FAKE_WEATHER_DB[key]
        return (
            f"Weather in {city_name.title()}: "
            f"{w['condition']}, Temp: {w['temp']}, Humidity: {w['humidity']}"
        )
    # Unknown city — return random mock data
    temp = random.randint(10, 38)
    return f"Weather in {city_name.title()}: Partly cloudy, Temp: {temp}°C, Humidity: 65%"


# ── 2. LLM Setup (using Mistral free tier) ───────────────────────────────────
llm = ChatMistralAI(model="open-mistral-7b", temperature=0.5)


# ── 3. Create Agent (LangChain 1.x — LangGraph-based) ────────────────────────
agent = create_agent(
    model=llm,
    tools=[get_current_weather],
    system_prompt="You are a helpful weather assistant. Use the get_current_weather tool to answer weather questions.",
)


# ── 4. Run the Agent ──────────────────────────────────────────────────────────
queries = [
    "What is the weather in London?",
    "How is the weather in Mumbai and Tokyo?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"User: {query}")
    inputs = {"messages": [{"role": "user", "content": query}]}
    
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            for msg in values.get("messages", []):
                content = getattr(msg, "content", "")
                if content:
                    print(f"[{node}]: {content}")

print(f"\n{'='*60}\nDone!")
