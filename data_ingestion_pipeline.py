import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
def load_documents(docs_path="Docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=40):
    print(f"Spliting the documnets into chunks of {chunk_size}")

    split = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap =chunk_overlap
    )

    chunks = split.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"chunk: {i+1}")
            print(chunk.page_content)
            print(len(chunk.page_content))
            print("-"*50)
    return chunks

def vector_store_config(chunks, persistent_directory = "db/chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
         model_name="all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_documents(
        persist_directory=persistent_directory,
        documents=chunks,
        embedding=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}    
        )
    return vectordb    
def main():
    print("Imports done!")
    docs_path ="Docs" 
    persistent_directory = "db/chroma_db"
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectordb._collection.count()} documents")
        return vectordb
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    #load the docs
    documents = load_documents(docs_path)
    print(len(documents))
    #splitting the documnets into chunks
    chunks = split_docs(documents)
    print(len(chunks))
    #making the vector store of the embeddings of the tokens and storing the in chroma db
    vectordb = vector_store_config(chunks,persistent_directory)

 

if __name__=="__main__":
    main()    