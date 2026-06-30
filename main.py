import psycopg  #-> Function->table_connection()
from langchain_postgres import PostgresChatMessageHistory #->Function->table_creation()
from pydantic import BaseModel, Field #-> Function->chatrequest()
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings  #-> Function ->vecrtor_store()
import os 
import pandas as pd 
import asyncio
import uuid
import uvicorn
from langchain_core.documents import Document #-> Function ->vecrtor_store()
from langchain_chroma import Chroma #-> Function ->vecrtor_store()
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #-> variable -> prompt_setup
from langchain_ollama import ChatOllama #-> variable -> model_setup
from langchain_core.runnables.history import RunnableWithMessageHistory # variable -> chat_setup
from langchain_core.runnables import RunnablePassthrough #-> variable -> rag_setup

def tabel_connection():
    return psycopg.connect(
        dbname = "ai_chat_db",
        user = "postgres",
        password = "s_o_n_m_o_n2004",
        host = "localhost",
        port ="5432",
        autocommit = True
                           )
connection = tabel_connection()

def tabel_creation():
    PostgresChatMessageHistory.create_tables(connection, "store_message")
tabel_creation()

class chatrequest(BaseModel):
    user_input : str = Field(default = "User input")
    user_login : str = Field(default =  "User Login")

app = FastAPI(title = "FAQ BOT")

persist_direct = "./chroma_db" #vector storage location 
faq_file = "BankFAQs.csv" #FAQ rules file

def vector_storage():
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    if os.path.exists(persist_direct):
        vector_store = Chroma(
            persist_directory= persist_direct,
            embedding_function = embeddings
        )
    else:
        with open(faq_file,"r") as file:
            data = pd.read_csv(file)
            docs = []
            for idx,i in data.iterrows():
                sections = Document(
                page_content = f"Question : {i['Question']}, \n Answers:{i['Answer']}",
                metadata = {"source":"faq_data"}
            )
                docs.append(sections)

        vector_store = Chroma.from_documents(documents=docs,
                                             embedding=embeddings,
                                             persist_directory=persist_direct)
    return vector_store.as_retriever(search_kwargs ={"k":2})

get_vector = vector_storage()

def search_faq_doc(dataset):
    user_question = dataset["input"]
    store_doc = get_vector.invoke(user_question)
    return "\n\n------------------\n\n".join(doc.page_content for doc in store_doc)

prompt_setup = ChatPromptTemplate.from_messages([
    ("system" , ("You are a Bank FAQ bot,"
    "answer the questions raised by the users using the Context below and in case of questions why don't have access or information on reply with NO DATA  "
    "Context: {context}")),
    MessagesPlaceholder(variable_name="history"),
    ("user","{input}")])

model_setup = ChatOllama(model="deepseek-r1:8b")

rag_setup = (RunnablePassthrough.assign(
    context = search_faq_doc) | 
    prompt_setup | 
    model_setup)

def get_user_history(session_id : str):
    try:
        valid_id = str(uuid.UUID(session_id))
    except:
        valid_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))
    
    return PostgresChatMessageHistory(
        "store_message",
        valid_id,
        sync_connection = connection
    )


chat_setup = RunnableWithMessageHistory(
    rag_setup,
    get_session_history= get_user_history,
    input_messages_key= "input",
    history_messages_key= "history"
)

@app.post("/chat")
async def chat_window(request : chatrequest):

    llm_response = await asyncio.to_thread(
        chat_setup.invoke,
        {"input": request.user_input},
        config = {"configurable" : {"session_id" : request.user_login}}
    )
    return {"Response" : llm_response.content}

if __name__ == "__main__":
    uvicorn.run("main:app",host= "127.0.0.1", port =8000,reload=True)