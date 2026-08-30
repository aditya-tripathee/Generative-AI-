from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document 

docs = [
    Document(page_content="Artifical Intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction. "
    , metadata={"name":"Aditya", "role":"ML Engineer"}),
    Document(page_content="Deep Learning (DL) is a subfield of machine learning based on artificial neural networks with representation learning. It is a part of broader machine learning methods using artificial neural networks and statistical learning methods, used in big data mining and analysis. DL has shown significant success in areas like image recognition, natural language processing, and speech recognition." 
    , metadata={"name":"Aditya", "role":"ML Engineer"}),
    Document(page_content= """Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and human language. It involves developing algorithms and models that enable computers to understand, interpret, and generate human language. NLP has applications in various domains, including chatbots, machine translation, sentiment analysis, and text summarization. Recent advancements in deep learning have significantly improved NLP tasks, leading to more accurate and context-aware language understanding and generation.""", metadata={"name":"Aditya", "role":"ML Engineer"}),
    Document(page_content=""" Computer Vision (CV) is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos. From an engineering perspective, it seeks to automate tasks that the human visual system can do. Computer Vision tasks include methods for acquiring, processing, analyzing, and understanding digital images, and extracting high-dimensional data from them. """, metadata={"name":"Aditya", "role":"ML Engineer"}),
    Document(page_content=""" Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a cumulative reward. It is inspired by behavioral psychology and has applications in robotics, game playing, and resource management. Key components of RL include the agent, environment, state, action, and reward. """, metadata={"name":"Aditya", "role":"ML Engineer"})
    ]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db" #directory where the vectorstore will be stored
)

# print("DB created successfully")

# result = vectorstore.similarity_search("What is AI?")
# print(result)



# used retrivers to get specific chunks of documents

result = vectorstore.similarity_search("What is AI?", k=2)
# print(result.page_content)
# print(result.metadata)


for r in result:
    print(r.page_content)
    print(r.metadata)

RETRIVER = vectorstore.as_retriever(search_kwargs={"k":2})

result = RETRIVER.get_relevant_documents("What is AI?")

for r in result:
    print(r.page_content)
    print(r.metadata)


    # creating vector database from list of documents

    vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)

    # retrive documents from the vector database

    retriver = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":2, "fetch_k":5}
    )
    
    result = retriver.get_relevant_documents("What is AI?")

    for r in result:
        print(r.page_content)
        print(r.metadata)