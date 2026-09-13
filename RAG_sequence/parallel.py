from dotenv import load_dotenv
load_dotenv()

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 2. Model
model = ChatMistralAI(model="open-mistral-7b")

# 3. Output parser
output_parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ]
)
detailed_prompt = ChatPromptTemplate.from_messages( 
    [
        ("system", "You are a helpful assistant. Give detailed answers."),
        ("user", "{question}"),
    ]
)
# formatted_short = short_prompt.format(question="What is the capital of France?")
# formatted_detailed = detailed_prompt.format(question="What is the capital of France?")

chain = RunnableParallel(
    short=RunnableLambda(lambda x: x["short"]) | short_prompt | model | output_parser,
    detailed=RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | output_parser,
)

# Run the chain
response = chain.invoke(
    {
        "short": "What is the capital of France?",
        "detailed": "What is the capital of Kathmandu Nepal?"
    }
    )
print("Response:", response)
print("Type of response:", type(response))