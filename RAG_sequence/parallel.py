from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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
    short=short_prompt | model | output_parser,
    detailed=detailed_prompt | model | output_parser,
)

# Run the chain
response = chain.invoke({"question": "What is the capital of France?"})
print("Response:", response)
print("Type of response:", type(response))  