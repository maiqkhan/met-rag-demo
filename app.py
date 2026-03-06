import ollama 
import os
from qdrant_client import models, QdrantClient
from fastapi import FastAPI
import openai

## Vector Search component
qd_client = QdrantClient("http://localhost:6333")

ollama_client = ollama.Client(host="http://localhost:11434")
 
openai_client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])


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

# question = "Which paintings in the MET museum are there about Jerusalem?"

# search_results = vector_search(question)

# for result in (search_results):
#     print("\n")
#     print(result)
#     print("\n")

def build_prompt(query, search_results):
    
    context = ""

    for doc in search_results:
        #print(doc['artwork_text'])
        #print('\n')
        context = context + f"artwork_description: {doc['artwork_text']}\nimage_url: {doc['artwork_image_url']}\ngallery_link: {doc['artwork_gallery_link']}\n"

    prompt = f"""
        You are an AI assistant that answers user questions about artworks in the European Paintings collection at the Metropolitan Museum of Art. 
        You will be given context information from the museum's knowledge base. Use ONLY this context to answer the user's question. 
        If the answer cannot be found in the context, say "I could not find that information in the collection. Could you elaborate further on your question?" and then explain what is confusing you about the question. 
        Do not hallucinate or make up facts.

        Guardrails:
        - Answer in clear, concise, and user-friendly language. 
        - If available, include the title, artist, date, medium, and link to the official museum page. 
        - If images are provided in the context, return their URLs so they can be displayed. 
        - Keep the response factual and grounded in the provided context.

        
        User Question:
        {query}

        Context:
        {context}


""".strip()
    
    
    return prompt

def rag(query):
    print("Looking for relevant documents!")
    search_results = vector_search(query)
    print("Found relevant documents!")
    prompt = build_prompt(query, search_results)
    print("Answering Question\n")
    # answer = ollama_client.generate(
    #     model="tinyllama",
    #     prompt=prompt
    # )
    answer = openai_client.chat.completions.create(
        model=os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    #return answer["response"]
    return answer.choices[0].message.content.strip()

question = "Which gallery room is Jerusalem from the Mount of Olives by Charles-Théodore Frère located in the MET Museum in New York? Could you provide a link to the gallery location as well?"

answer  = rag(question)

print(answer)

# app = FastAPI()


# @app.get("/ask")
# def ask(question: str):
#     answer = rag(question)
#     return {"answer": answer}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
