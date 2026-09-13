from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{question}"),
    ]
)

# 2. Model
model = ChatMistralAI(model="open-mistral-7b")

# 3. Output parser
output_parser = StrOutputParser()

# 4. Chain using LCEL pipe operator (prompt | model | parser)
chain = prompt_template | model | output_parser

# Run the chain
response = chain.invoke({"question": "What is the capital of France?"})

print("Response:", response)
print("Type of response:", type(response))