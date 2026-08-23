
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# # 👇 .env load karo
# load_dotenv()

# # 👇 token automatically env se uth jayega
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
# )

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("Who is Birendra Shah?")
# print(response.content)


# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# # Load .env from Basic folder
# env_path = Path(__file__).resolve().parent.parent / ".env"
# load_dotenv(dotenv_path=env_path)

# # Ensure HF token is accessible to Hugging Face client
# hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
# if hf_token:
#     os.environ["HF_TOKEN"] = hf_token
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-72B-Instruct",
#     huggingfacehub_api_token=hf_token
# )

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("Who is Birendra Bir Bikram Shah?")
# print(response.content)



from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)


response = chat_model.invoke("Tell me something about Birendra Bir Bikram Shah.")
print(response.content)