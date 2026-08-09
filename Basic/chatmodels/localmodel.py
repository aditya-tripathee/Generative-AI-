# from langchain_huggingface import ChatHuggingFace, HugginggingFacePipeline
# from huggingface_hub import hf_hub_download

# llm = HuggingFaceEndpoint.from_model_id(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("Who is Birendra Bir Bikram Shah?")
# print(response.content)

# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# # ✅ local pipeline
# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     max_new_tokens=200
# )

# # ✅ convert to LangChain
# llm = HuggingFacePipeline(pipeline=pipe)

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("Who is Birendra Bir Bikram Shah?")
# # print(response.content)


# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     device_map="auto"   # uses GPU if available
# )



# llm = HuggingFacePipeline(pipeline=pipe)

# model = ChatHuggingFace(llm=llm)

# response = model.invoke(
#     "User: Who is Birendra Bir Bikram Shah?\nAssistant:",
#     max_new_tokens=150,
#     temperature=0.6,
#     do_sample=True
# )

# print(response.content)





from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# ✅ Create pipeline (NO generation params here)
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"   # uses GPU if available
)


# ✅ Convert to LangChain
llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# ✅ Proper chat-style prompt + generation params here
response = model.invoke(
    "User: Who is Birendra Bir Bikram Shah?\nAssistant:",
    max_new_tokens=150,
    temperature=0.6,
    do_sample=True
)

print(response.content)