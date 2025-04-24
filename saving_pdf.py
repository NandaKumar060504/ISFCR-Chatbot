from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.vectorstores import FAISS

loaders = DirectoryLoader('C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\pdfs', glob="*.pdf", loader_cls=PyPDFLoader)

docs= loaders.load()

textsplit = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
splits = textsplit.split_documents(docs)

embeddings = OllamaEmbeddings(model='nomic-embed-text')
db = FAISS.from_documents(splits,embeddings)
db.save_local("C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\faiss")