import arxiv
from langchain_community.retrievers import ArxivRetriever 

# Compatibility patch for arxiv>=2.0
if not hasattr(arxiv.Search, "results"):
    arxiv.Search.results = lambda self: arxiv.Client().results(self)

retriever = ArxivRetriever(
    load_max_docs=5,
    doc_content_type="summary", 
    top_k_results=10,
)


#query arixv 
docs = retriever.invoke("what is the capital of nepal")


print(len(docs)) 
print(docs[0])


