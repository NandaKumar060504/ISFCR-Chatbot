from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings 

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Loading environment variables from .env file
load_dotenv() 

#Document
class Document():
        def __init__(self,text,source):
            self.page_content=text
            self.metadata={"source":source}


# GROQ_API_KEY="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"
# groq_api_key = os.environ["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"]

# Initializing GROQ chat with provided API key, model name, and settings


from langchain_community.chat_models import ChatOllama
llm_local= ChatOllama(model="mistral")

llm_groq = ChatGroq(
            groq_api_key="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", model_name="mixtral-8x7b-32768",
                         temperature=0.2)
#########################################
import PyPDF2
import os
file_paths = [
  'V1_QR.pdf',
  'V2_QR.pdf',
  'V3_QR_Conferences.pdf',
  'V3_QR_Journals.pdf'
]

texts = []
text=""
for i in file_paths:
  text=""
  pdfFile = open(i,'rb')
  Reader = PyPDF2.PdfReader(pdfFile)


  for page in range(len(Reader.pages)):
    pageObj = Reader.pages[page]
    text += pageObj.extract_text()
  texts.append(text)

print(texts)

################################


  # Chunk text
text_splitter = CharacterTextSplitter(chunk_size=100,
                                        chunk_overlap=0)

  #lets try inserting own documents
documents = [Document(text,"source") for text in texts]

chunked_documents_text = text_splitter.split_documents(documents)


  # Load chunked documents into the FAISS index
# db = FAISS.from_documents(chunked_documents_text,
#                             embeddings = OllamaEmbeddings(model="nomic-embed-text"))
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
db = FAISS.from_documents(chunked_documents_text,
                            FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

retriever = db.as_retriever()

##########################

import langchain_core
# # Prepare Document object (modify based on actual requirements)
# document = langchain_core.documents.base.Document(page_content=text)
# # document.text = text  # Assuming there's a "text" attribute in the Document class
# docs_transformed.append(document)
# print(docs_transformed)

# chunked_documents=text_splitter.split_documents(docs_transformed)

prompt_template = """
### [INST] Instruction: You are a Chat Bot assistant. You must behave like a PESU Research Review assistant. You should be able to answer any question with maximum accuracy providing valid information. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

from langchain.prompts import PromptTemplate
# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

llm_chain = LLMChain(llm=llm_local, prompt=prompt)

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

result = rag_chain.invoke("List all the publications of Prasad Honnavalli in the year 2022 from Research Volume 2. ")
 
print(result['context'] , result['text'])

# message_history = ChatMessageHistory()

    
    
#     # Memory for conversational context
# memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

#     # Create a chain that uses the Chroma vector store
# chain = ConversationalRetrievalChain.from_llm(
#         llm=llm_local,
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),          
#         memory=memory,
#         return_source_documents=True,
#     )


def chat(chat_history, user_input):

  result = rag_chain.invoke(user_input)

  response = ""
  for letter in ''.join(result['text']): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
      response += letter + ""
      yield chat_history + [(user_input, response)]


import gradio as gr
with gr.Blocks() as demo:
    with gr.Tab("PESU Research Review Bot"):
#          inputbox = gr.Textbox("Google Scholar PESU ISFCR Bot....")
          chatbot = gr.Chatbot()
          message = gr.Textbox ("Type your queries here!")
          message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch(share= True)