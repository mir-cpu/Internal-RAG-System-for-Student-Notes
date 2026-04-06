import os
import json,gc,time
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

docs_path="Docs"
persistent_directory = "db/chroma_db"
config_path = "Config/config_path.json"
db_config_path = "db/db_config.json"

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
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your notes.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_docs(documents, chunk_size, chunk_overlap):
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

#utility functions

def load_config():
     with open(config_path,"r") as file:
        return json.load(file)

def config_changed():
    if not os.path.exists(db_config_path):
        return True
    with open(db_config_path,"r") as file:
        old_config = json.load(file)

    new_config = load_config()
    return new_config!=old_config

def save_config():
    new_config=load_config()
    with open(db_config_path,"w") as file:
        json.dump(new_config,file,indent =2)


def delete_db():
     #deletes the existing db from memory, prevents access denial errors
    gc.collect()
    global vectordb
    vectordb = None
    if os.path.exists(persistent_directory):
        shutil.rmtree(persistent_directory)
        print("Old DB deleted")                
 

def main():
    print("Imports done!")
    #paths for documents, persistence and configurations versions (tracking version, chunk sizes and overlaps)
    
    if config_changed():
        delete_db()
        #load the docs
        documents = load_documents(docs_path)
        config = load_config()
        new_chunk_size = int(config["chunk_size"])
        new_chunk_overlap = int(config["chunk_overlap"])

        #splitting the documnets into chunks
        chunks = split_docs(documents,new_chunk_size,new_chunk_overlap)
        print(len(chunks))
        #making the vector store of the embeddings of the tokens and storing the in chroma db
        vectordb = vector_store_config(chunks,persistent_directory)
        save_config()

    else:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectordb._collection.count()} documents")
        return vectordb
    
 

if __name__=="__main__":
    main()    