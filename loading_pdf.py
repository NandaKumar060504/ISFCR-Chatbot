from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

data_path = "C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\faiss"
embeddings = OllamaEmbeddings(model='nomic-embed-text')
db =  FAISS.load_local(data_path,embeddings,allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k":50})
question = "In which Conference of the paper titled Indoor Violence Detection using Lightweight Transformer Model "
retrieved_docs = retriever.invoke(question)
formatted_prompts = format_docs(retrieved_docs)
formatted_prompt = f"Question: {question}\n\nContext: {formatted_prompts}"
with open("prompt.txt","w") as f:
    f.write(formatted_prompt)
response = ollama.chat(model='mistral:instruct', messages=[{'role': 'user', 'content': formatted_prompt}],)
print(response['message']['content'])