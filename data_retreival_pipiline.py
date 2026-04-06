from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
         "k": 3,
         "score_threshold": 0.5 
        }
)

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in relevant_docs])

prompt_template = f"""
    you have to answer the query ONLY based on context,
    if answer exists explicitly, return it directly and do not infer beyond context.
    If you do not know the answer, say , the query cannot be resolved in case of no information or context about the query asked.
    context:
    {context}
    user_question:
    {query}
    """
ai_model = Ollama(model="llama2:7b")

#invoke the llm
response = ai_model.invoke(query)

# print(f"User Query: {query}")
# # Display results
# print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

print(f"""the response is : {response}""")
# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"