from dotenv import load_dotenv
load_dotenv()

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


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

# 4. Chain with RunnablePassthrough
# RunnablePassthrough passes the input through unchanged.
# Combined with RunnableParallel, you can keep the original input
# alongside the generated answer in the final output dict.
chain = RunnableParallel(
    question=RunnablePassthrough(),          # passes {"question": "..."} unchanged
    answer=prompt_template | model | output_parser  # generates the answer
)

# Run the chain
response = chain.invoke({"question": "What is the capital of France?"})

print("Original question :", response["question"])
print("Model answer      :", response["answer"])
print("Type of response  :", type(response))


