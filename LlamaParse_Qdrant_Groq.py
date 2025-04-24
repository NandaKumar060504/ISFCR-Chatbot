import os
import llama_index
from llama_index_client import RetrieveResults
from llama_parse import LlamaParse
import llama_index.vector_stores.qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv() 

#Loading Llama Parse API Key
llamaparse_api_key = os.getenv("Llama__Cloud_API_KEY")

#Llama parse for PDF
# llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("LLm_ISFCR/GroqPDFFastChatbot-main/PESU Research Review Vol 3 - Abstract Version (1).pdf")

import pickle
# Define a function to load parsed data if available, or parse if not
def load_or_parse_data():
    data_file = "./LLm_ISFCR/GroqPDFFastChatbot-main/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("LLm_ISFCR\GroqPDFFastChatbot-main\Research_Review_3-compressed.pdf")
        #llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/presentation.pptx")
        # llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data(["./data/presentation.pptx", "./data/uber_10q_march_2022.pdf"])

        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data
# Call the function to either load or parse the data
llama_parse_documents = load_or_parse_data()


print(llama_parse_documents)

# llama_parse_documents[0].text[:100]

######## QDRANT ###########

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

import qdrant_client 

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

######### FastEmbedEmbeddings #############

# by default llamaindex uses OpenAI models

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

""" embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    #model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
) """

#### Setting embed_model other than openAI ( by default used openAI's model)
from llama_index.core import Settings

Settings.embed_model = embed_model

######### Groq API ###########

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="mixtral-8x7b-32768", api_key=groq_api_key)
#llm = Groq(model="gemma-7b-it", api_key=groq_api_key)

######### Ollama ###########

#from llama_index.llms.ollama import Ollama  # noqa: E402
#llm = Ollama(model="llama2", request_timeout=30.0)

#### Setting llm other than openAI ( by default used openAI's model)
Settings.llm = llm

def Qdrant():

    client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)
    vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents=llama_parse_documents, storage_context=storage_context, show_progress=True)

    #### PERSIST INDEX #####
    index.storage_context.persist()

    #storage_context = StorageContext.from_defaults(persist_dir="./storage") 
    
    #index = load_index_from_storage(storage_context)

    # create a query engine for the index
    query_engine = index.as_query_engine()
    # query = "List all the publications this document contains"
    # response = query_engine.query(query)
    # print(response)
    # Assuming you have a retriever object (created using FaissVectorStore.as_retriever() or similar)
    retrieve = FAISS.as_retriever(query_engine)  # Replace 'db' with your actual store object

    return retrieve

# def rag_chain(human):

#     system = "You are an ISFCR Chat Assistant. You must understand the entire document content and you must answer every question in a detailed manner by giving the most accurate output without hallucinating. Do not consider the external Reference publications while answering queries."
#     human = "{text}"
#     prompt = {"system": system, "human": human}
    
#     query = human
#     # Create a query engine from the retriever
#     retrieval_query_engine = retrieve.as_query_engine()
#     query_response = retrieval_query_engine.query(query)
#     retrieved_response = retrieve.invoke(human)
#     print(retrieved_response)
    # response = llm.run(prompt)
    # return response
    
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# Combine system intro and user query into a single prompt
# prompt = f"{system}\n{human}"
    

# response = query_engine.query(query)
# query = "List all the publications this document contains"
# response = query_engine.query(query)
# print(response)

# chain = prompt | llm
# chain.invoke({"text": "List all the research publications under Prasad."})
# chain.invoke(response)
    
# Initialize message history for conversation
# message_history = ChatMessageHistory()
    
# # Memory for conversational context
# memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

# # Create a chain that uses the Chroma vector store
# chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         chain_type="stuff",
#         retriever=as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )

# import gradio as gr
# # Gradio interface
# iface = gr.Interface(
#     fn=rag_chain,
#     inputs=["text"],
#     outputs="text",
#     title="RAG Chain Question Answering",
#     description="Enter a URL and a query to get answers from the RAG chain."
# )

# # # Launch the app
# iface.launch(share=True)

#Initialization:

# Imports necessary libraries (Llama-Index, Langchain, Chainlit, etc.).
# Loads environment variables from a .env file.
# Sets up Llama-Parse for PDF parsing.
# Defines a function to load or parse PDF data efficiently.
# Loads parsed documents (either from a file or by parsing a PDF).
# Configures Qdrant vector store for indexing.
# Creates embeddings, LLM model, and vector store.
# Persists the index for later use (optional).

# Chat Start Handler (@cl.on_chat_start):

# Prompts the user to upload a PDF file.
# Processes the uploaded PDF to extract text.
# Splits text into chunks for indexing (optional, commented out).
# Creates a message history and conversational memory for context.
# Creates a vector store and a retrieval chain.
# Informs the user that the system is ready for questions.
# Stores the chain in the user session for later use.


