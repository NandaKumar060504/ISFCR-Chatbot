from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import os
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder
from re import search
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask, render_template, request, jsonify

def load_web(webpaths):
    """
    Args: webpaths
    What it does : WebBaseLoader loads web content
    What it returns: Contents of web page
    """
    web_loader  = WebBaseLoader(web_paths=(webpaths))
    web = web_loader.load()
    return web

def load_pdf(path):
    """
    Args: pdf paths
    What it does : PyPDFLoader loads pdf content
    What it returns: Contents of pdf
    """
    pdf_loader = PyPDFLoader(path)
    docs = pdf_loader.load()
    return docs

def split_documents(docs):
    """
    Args: document content
    What it does : Splits the document content
    What it returns: Split content
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits

def load_embeddings(repo_id):
    """
    Args: repo_id
    What it does : creates embeddings
    What it returns: Embedding model
    """
    embeddings = HuggingFaceEmbeddings(model_name=repo_id)
    return embeddings

def create_vec_store(docs,embeddings,collection_name,directory):
    """
    Args: splitted content, embedding model, collection name, database directory
    What it does : Creates a vector store
    What it returns: Returns a vector store
    """
    client_settings = chromadb.config.Settings(is_persistent=True,persist_directory=directory,anonymized_telemetry=False)
    vec_store = Chroma.from_documents(docs,embeddings,collection_name=collection_name,collection_metadata={"hnsw":"cosine"},persist_directory = directory)
    return vec_store

def create_retriever(vectorstore):
    """
    Args: vectorstore
    What it does : Retrieves content from vectorstore
    What it returns: retriever
    """
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 7, "include_metadata": True})
    return retriever

def merge_retrievers(retrievers):
    """
    Args: retrievers
    What it does : Creates a merger retriever 
    What it returns: retriever
    """
    lotr = MergerRetriever(retrievers=retrievers)
    return lotr

def create_filter(embeddings):
    """
    Args: embeddings
    What it does : filters out the embeddings
    What it returns: created_filter
    """
    created_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    return created_filter

def create_reorder():
    
    reordering = LongContextReorder()
    return reordering

def create_pipeline(transformers):
    """
    Args: Transformers
    What the function does: creates a document compressor pipeline by taking filters as transformers
    what the functions return: It returns compressed document pipeline"""
    pipeline = DocumentCompressorPipeline(transformers=transformers)
    return pipeline

def create_compression_retriever(base_compressor,base_retriever):
    """
    Args: Transformers
    What the function does: creates a document compressor pipeline by taking filters as transformers
    what the functions return: It returns compressed document pipeline"""
    compression_retriever_reordered = ContextualCompressionRetriever(base_compressor=base_compressor, base_retriever=base_retriever,search_kwargs={"k": 7, "include_metadata": True})
    return compression_retriever_reordered

def load_llm(model_path):
    """
    Args: model path
    What the function does: Loads the llm by taking the path where the model is located
    what the functions return: It returns llm"""
    llm = LlamaCpp(streaming=True,model_path=model_path,max_tokens = 1500,temperature=0.75,top_p=1,gpu_layers=1,stream=True,verbose=True,n_threads = int(os.cpu_count()/2),n_ctx=4096)
    return llm


def create_prompt_template(template):
  
    prompt_temp = PromptTemplate.from_template(template)
    return prompt_temp


def create_ret_chain(llm,retriever,prompt_temp):
 
    qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever = retriever,
      return_source_documents = True,
      chain_type_kwargs={"prompt": prompt_temp}
    )
    return qa



Flag = False
DB = "c:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\new_folder\\db"
if not(Flag):
        print("inside if block")
        web_paths=([
        "https://www.isfcr.pes.edu","https://www.isfcr.pes.edu/about","https://www.isfcr.pes.edu/team","https://www.isfcr.pes.edu/research","https://www.isfcr.pes.edu/courses","https://www.isfcr.pes.edu/testimonials","https://www.isfcr.pes.edu/events","https://www.isfcr.pes.edu/bug-bounty","https://www.isfcr.pes.edu/resources","https://www.isfcr.pes.edu/team/prasad-b-honnavalli","https://www.isfcr.pes.edu/team/preet-kanwal","https://www.isfcr.pes.edu/team/indu-radhakrishnan","https://www.isfcr.pes.edu/team/vadiraja-acharya","https://www.isfcr.pes.edu/team/vanitha-g","https://www.isfcr.pes.edu/team/dr.-ashok-kumar-patil","https://www.isfcr.pes.edu/team/dr.-s.s.-iyengar","https://www.isfcr.pes.edu/team/dr-nagasundari-s","https://www.isfcr.pes.edu/team/revathi","https://www.isfcr.pes.edu/team/dr.-shruti-jadon","https://www.isfcr.pes.edu/team/sapna-v-m","https://www.isfcr.pes.edu/team/dr.-radhika-m.-hirannaiah","https://www.isfcr.pes.edu/team/sushma-e","https://www.isfcr.pes.edu/team/charanraj-b-r","https://www.isfcr.pes.edu/team/dr.-adithya-b","https://www.isfcr.pes.edu/team/dr.-roopa-ravish","https://www.isfcr.pes.edu/team/pavan-a-c","https://www.isfcr.pes.edu/courses/computer-networks","https://www.isfcr.pes.edu/courses/internet-of-things","https://www.isfcr.pes.edu/courses/fundamentals-of-augmented-and-virtual-reality","https://www.isfcr.pes.edu/courses/computer-network-security","https://www.isfcr.pes.edu/courses/cyber-forensics","https://www.isfcr.pes.edu/courses/wireless-network-communication","https://www.isfcr.pes.edu/courses/topics-in-computer-and-network-security","https://www.isfcr.pes.edu/courses/software-security","https://www.isfcr.pes.edu/courses/cyber-security-essentials","https://www.isfcr.pes.edu/courses/applied-cryptography","https://www.isfcr.pes.edu/courses/computer-networks-laboratory","https://www.isfcr.pes.edu/courses/applied-cryptography-(ug)","https://www.isfcr.pes.edu/courses/advanced-computer-networks","https://www.isfcr.pes.edu/courses/blockchain","https://www.isfcr.pes.edu/courses/information-security","https://www.isfcr.pes.edu/courses/virtual-reality-and-its-applications","https://www.isfcr.pes.edu/courses/cryptography","https://www.isfcr.pes.edu/courses/topics-in-cryptography","https://www.isfcr.pes.edu/post/two-tier-securing-mechanism-against-web-application-attacks-project-presentation-to-citrix-ceo","https://www.isfcr.pes.edu/post/preventing-community-outbreaks-from-silent-spreaders-of-covid-19-virus","https://www.isfcr.pes.edu/post/have-you-thought-of-cybercrimes-during-the-covid-19-pandemic","https://www.isfcr.pes.edu/post/homomorphic-encryption","https://www.isfcr.pes.edu/post/do-you-know-about-the-underlying-system-attacked-by-stuxnet-worm-in-iran","https://www.isfcr.pes.edu/resources/page/2","https://www.isfcr.pes.edu/post/isfcr-gets-paper-published-1",
        "https://www.isfcr.pes.edu/post/so-now-what-student-stories-from-graduation",
        "https://www.isfcr.pes.edu/post/why-you-should-be-making-the-most-of-student-study-groups",
        "https://www.isfcr.pes.edu/resources/hashtags/facts",
        "https://www.isfcr.pes.edu/resources/hashtags/trivia",
        "https://www.isfcr.pes.edu/resources/hashtags/studygroup",
        "https://www.isfcr.pes.edu/resources/hashtags/campus",
        "https://www.isfcr.pes.edu/resources/hashtags/educatio"
        ]
        )
        web=load_web(web_paths)
        path_1 = "C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\pdfs\\Research_Vol_2021.pdf"
        pdf_1 = load_pdf(path_1)
        path_2 = "C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\pdfs\\Research_Vol_2022.pdf"
        pdf_2 = load_pdf(path_2)
        path_3 = "C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\pdfs\\Research_Vol_2023.pdf"
        pdf_3 = load_pdf(path_3)
        pdf = pdf_1+pdf_2+pdf_3
        pdf_split = split_documents(pdf)    
        web_split = split_documents(web)
        repo_id_1 = "BAAI/bge-base-en-v1.5"
        repo_id_2 = "sentence-transformers/all-MiniLM-L6-v2"
        repo_id_3="sentence-transformers/all-mpnet-base-v2"
        embedding_1 = load_embeddings(repo_id_1)
        embedding_2 = load_embeddings(repo_id_2)
        embedding_3=load_embeddings(repo_id_3)
        pdf_vectorstore = create_vec_store(pdf_split,embedding_1,"pdf_collection",DB)
        web_vectorstore = create_vec_store(web_split,embedding_2,"web_collection",DB)
        pdf_retriever=create_retriever(pdf_vectorstore)
        web_retriever=create_retriever(web_vectorstore)
        lotr=merge_retrievers([pdf_retriever,web_retriever])
        embedding_filter=create_filter(embedding_3)
        reordering=create_reorder()
        pipeline=create_pipeline([embedding_filter,reordering])
        compressor_retriever_reordering=create_compression_retriever(pipeline,lotr)
        llm=load_llm("C:\\Users\\Aiml cse\\Downloads\\zephyr-7b-beta-q4_k_m.gguf")
        template="""Answer the question based only on the following context. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Do not repeat any points in the answer. 
        Do not manipulate names of papers, authors or anything just give the same ones if present. 
        Do not mention anything like "From the given context". 
        Also provide web links supporting the answer if necessary and if they are mentioned.
        Also dont mention things like page numbers,etc. which are irrelevant to the question.
                    
                    
                    Context: {context}
                    Question: {question}
                    Helpful Answer:"""

        prompt_temp=create_prompt_template(template)
        Flag = True

def llm_chain(message):
    
    qa=create_ret_chain(llm,compressor_retriever_reordering,prompt_temp)


    results = qa(message)
    
    return results['result']

    #print(results["source_documents"])

# app = Flask(__name__)
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/chat', methods=['GET', 'POST'])
# def chat():
#     message = request.form['msg']
#     return llm_chain(message)

# if __name__ == "__main__":
#      app.run() 
# For example List all the papers authored by Prasad B Honnavalli ,the answer as such The papers authored by Prasad B Honnavalli are : 
                    # 1) Managing Data Protection and Privacy on Cloud Satyavathi Divadari, J. Surya Prasad, Prasad Honnavalli Conference published in: Proceedings of 3rd International Conference on Recent Trends in Machine Learning, IoT, Smart Cities and Applications. Lecture Notes in Networks and Systems, vol 540. Springer, Singapore
                    # 2)Malicious Network Traffic Detection in Internet of Things Using Machine Learning Manjula Ramesh Bingeri, Sivaraman Eswaran, Prasad Honnavalli Conference published in: Proceedings of Data Analytics and Management. Lecture Notes in Networks and Systems, vol 572. Springer, Singapore. 
                    # 3) Cloud-based Network Intrusion Detection System using Deep Learning ,Archana, Chaitra H P, Khushi, Sivaraman, Prasad Honnavalli
                    # 4) Multiple Hashing Using SHA-256 and MD5 Gautham P. Reddy, Anoop Narayana, P. Karan Keerthan, B. Vineetha, Prasad Honnavalli


                    # Helper function for printing docs


# def pretty_print_docs(docs):
#     print(
#         f"\n{'-' * 100}\n".join(
#             [
#                 f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
#                 for i, d in enumerate(docs)
#             ]
#         )
#     )
     