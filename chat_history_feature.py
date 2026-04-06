from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time

# setting up the database , embedding model, llm model 
ai_model = Ollama(model="llama2:7b")
persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")
db  = Chroma(persist_directory=persistent_directory,embedding_function=embedding_model,
             collection_metadata={"hnsw:space":"cosine"})

chat_history=[]

#step 1 get the user question and ask the model to reformulate it
def conversation_with_user(user_input):

    if chat_history:

        messages = ([SystemMessage(content="Given the chat history and user question , rewrite this into a new question as a standalone and searchable question")]
        + chat_history + [HumanMessage(content=f"You asked : {user_input}")])
        
        start  = time.time()
        result = ai_model.invoke(messages)
        print("The time taken to reformulate the question ",time.time() - start)
        search_question = result.strip()
        print(f"Searching for -- {search_question}")
    else:
        search_question = user_input

    # step 2 look for the similarity check in vector store

    retriver = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":2,
            "score_threshold":0.4
        }
    )
    start = time.time()
    rel_docs = retriver.invoke(search_question)
    print("the time taken for fetching relevant documents : ", time.time()-start)

    for i, doc in enumerate(rel_docs):
         print(f"Dcoument retreived: {i}")
         print(f"content: {doc.page_content}")
    combined_prompt = f""" based on the follwing documents answer the user query{user_input}
        documents:
        {"\n".join([doc.page_content for doc in rel_docs])}
        Please provide a userful answer based on the information from these documents only.
        If you do not have enough information , say , that you do not know the answer.
        """
    messages = ([SystemMessage(content="you are a helpful assistant that answer students queries based on provided documents and conversation history")]
               +chat_history+[HumanMessage(content = combined_prompt)])

    #pdb.set_trace()
#ask the llm
    start = time.time() 
    result = ai_model.invoke(messages)

    print("the time taken by llm to get the answer: ", time.time()-start )
    print(type(result))
    # append the human and ai conversation in chat history
    chat_history.append(HumanMessage(content=str(user_input)))
    chat_history.append(AIMessage(content=result))

    return result



def ask_question_loop():
        while True:
             
            print(f"Hi welcome to smart notes, do you want to ask any questions..? or type quit ")
            user_input = input("clear your doubts..").lower()
            if(user_input=="quit"):
                 break
            output = conversation_with_user(user_input)
            print(output)
        # while(user_input!='quit'):
        #     conversation_with_user(user_input)
        
        # print("Thank you keep learning!")



def main():

    ask_question_loop()

main()    