from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH  = "../faiss"
embeddings = OllamaEmbeddings(model='nomic-embed-text')
db = FAISS.load_local(DB_FAISS_PATH,embeddings)