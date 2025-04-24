from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.vectorstores import FAISS

loaders = DirectoryLoader('C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes', glob="*.pdf", loader_cls=PyPDFLoader)

docs= loaders.load()
# print(docs)

textsplit = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
splits = textsplit.split_documents(docs)

embeddings = OllamaEmbeddings(model='nomic-embed-text')
db = FAISS.from_documents(splits,embeddings)
# db.save_local('/faiss')
retriever = db.as_retriever()
question = "Papers authored by Prasad Honnavalli?"
retrieved_docs = retriever.invoke(question)
formatted_prompt = f"Question: {question}\n\nContext: {retrieved_docs}"
response = ollama.chat(model='mistral:instruct', messages=[{'role': 'user', 'content': formatted_prompt}])
print(response)


# for loader in loaders:
#     docs.append(loader.load())
# print(len(docs))

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(doc for doc in docs)

# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# db = FAISS.from_documents(splits,embeddings)
    
# retriever = db.as_retriever()
# question = "what is rag?"
# retrieved_docs = retriever.invoke(question)

# formatted_prompt = f"Question:{ question} \n\n Context:{retrieved_docs}"
# response = ollama.chat(model="mistral:instruct",messages=[{'role': 'user', 'content': formatted_prompt}])
# print(response)
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def rag_chain(question):

#     retrieved_docs = retriever.invoke(question)
#     formatted_context = format_docs(retrieved_docs)
#     print(formatted_context)

#     # Create prompt from prompt template
  
#     formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    
#     response = ollama.chat(model='mistral:instruct', messages=[{'role': 'user', 'content': formatted_prompt}])

#     return response['message']['content']

# print(rag_chain({"question":"what is rag?"}))

# #from langchain.text_splitter import CharacterTextSplitter
# #from langchain.document_loaders import UnstructuredFileLoader
# from langchain.document_loaders import UnstructuredPDFLoader
# #from langchain.vectorstores.faiss import FAISS
# # from langchain.embeddings import OpenAIEmbeddings
# #import pickle
# import os
# print("Loading data...")
# pdf_folder_path = "C:\Users\Aiml cse\Desktop\ISFCR_Codes"
# print(os.listdir(pdf_folder_path))

# # Load multiple files
# # location of the pdf file/files. 
# loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]


# print(loaders)