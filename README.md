# met-rag-demo
A simple demo of a RAG application

## How RAG Works

**RAG (Retrieval-Augmented Generation)** is an AI architecture that enhances large language models by combining them with external knowledge retrieval. Instead of relying solely on the model's training data, RAG dynamically retrieves relevant information from a knowledge base to generate more accurate and contextual responses.

![RAG Architecture](images/RAG%20Image.png)

### Key Components

1. **Knowledge Base**: A collection of documents or data (in this case, Met Museum objects) that serves as the source of truth
2. **Embedding Model**: Converts text into numerical vectors that capture semantic meaning
3. **Vector Database**: Stores and indexes the embedded documents for efficient similarity search
4. **Retrieval System**: Finds the most relevant documents based on semantic similarity to the user's query
5. **Language Model**: Generates responses using both the retrieved context and the user's question

### The RAG Workflow

1. **Indexing Phase** (Setup)
   - Documents from the knowledge base are split into manageable chunks
   - Each chunk is converted into embeddings using an embedding model
   - Embeddings are stored in a vector database with their original text

2. **Query Phase** (Runtime)
   - User submits a question or query
   - The query is converted into an embedding using the same embedding model
   - Vector similarity search finds the most relevant document chunks
   - Retrieved chunks are combined with the user's query as context
   - The language model generates a response based on the retrieved information

3. **Response Generation**
   - The LLM receives both the user's question and the relevant retrieved context
   - It generates an answer grounded in the retrieved information
   - This reduces hallucinations and provides sources for verification

### Benefits of RAG

- **Up-to-date Information**: Knowledge base can be updated without retraining the model
- **Reduced Hallucinations**: Responses are grounded in actual retrieved documents
- **Transparency**: Retrieved sources can be cited and verified
- **Domain Specialization**: Works with specialized knowledge not in the LLM's training data
- **Cost-Effective**: No need to fine-tune large models on custom data

### Chunking


![Chunking Methods](images/Chunking%20Image.png)
