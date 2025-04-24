from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import ollama
# Passing URLs and PDF Document for Retrieval-Augmented Generation (RAG) to assist the LLM


# import nest_asyncio

# nest_asyncio.apply()

# attachment=""
# # Articles to index
# articles = [
#             " https://www.pgportal.gov.in/",
#             "https://www.pgportal.gov.in/Home/Faq",
#             "https://www.pgportal.gov.in/Home/AboutUs",
#             "https://www.pgportal.gov.in/Home/ContactUs",]

# # Scrapes the blogs above
# loader = AsyncChromiumLoader(articles)
# docs = loader.load()
# print(docs)
#     # Converts HTML to plain text
# html2text = Html2TextTransformer()
# docs_transformed = html2text.transform_documents(docs)
# print(docs_transformed)
#   # Chunk text

# text_splitter = CharacterTextSplitter(chunk_size=100,
#                                         chunk_overlap=0)



# chunked_documents=text_splitter.split_documents(docs_transformed)

# #   # Load chunked documents into the FAISS index
# db = FAISS.from_documents(chunked_documents,
#                             embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))

# retriever = db.as_retriever()

model_local = ChatOllama(model="mistral")

# 1. Split data into chunks
urls = ["https://www.isfcr.pes.edu/",
            "https://www.isfcr.pes.edu/about",
            "https://www.isfcr.pes.edu/team",
            "https://www.isfcr.pes.edu/research",
            "https://www.isfcr.pes.edu/courses",
            "https://www.isfcr.pes.edu/team/preet-kanwal"]


loader = AsyncChromiumLoader(urls)
docs = loader.load()
print(docs)
    # Converts HTML to plain text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
print(docs_transformed)

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
# text_splitter = CharacterTextSplitter(chunk_size=100,
#                                         chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_transformed)

# 2. Convert documents to Embeddings and store them
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
# )
db = FAISS.from_documents(doc_splits,
                            embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))

retriever = db.as_retriever()



# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)


def ask_ollama_rag(query):
    print(after_rag_chain.invoke(query))


while(1):
    query=input("query on isfcr")
    ask_ollama_rag(query)
    