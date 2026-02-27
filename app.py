import ollama 
from qdrant_client import models, QdrantClient

## Vector Search component
qd_client = QdrantClient("http://localhost:6333")

def vector_search(question):
    query_points = qd_client.query_points(
        collection_name='met-museum-artworks',  # database
        # sql query
        query=models.Document(
            text=question,
            model="jinaai/jina-embeddings-v2-small-en"
        ),

        #schema
        using="jina-small",
        #top 100 
        limit=5,
        #select * 
        with_payload=True
    )

    results = []

    for point in query_points.points:
        results.append(point.payload)

    return results

question = "Which paintings in the MET museum are there about Jerusalem?"

search_results = vector_search(question)

for result in (search_results):
    print("\n")
    print(result)
    print("\n")


## LLM Component

# ollama_client = ollama.Client(host="http://localhost:11434")

# answer = ollama_client.generate(
#     model="tinyllama",
#     prompt=f"Which paintings in the MET museume are there about Jerusalem?"
# )

# print(answer['response'])