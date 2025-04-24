from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain

data_path = "C:\\Users\\Aiml cse\\Desktop\\ISFCR_Codes\\faiss"
embeddings = OllamaEmbeddings(model='nomic-embed-text')
db =  FAISS.load_local(data_path,embeddings,allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":10})
llm = Ollama(model = "mistral:instruct")
template = """Answer the questions based on the below context.
If you don't know the answer, just say don't know answer.
{context}"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",template),
        ("user","{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm,prompt)
qa_chain = create_retrieval_chain(retriever,document_chain)
chain = (
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(qa_chain.pick("answer").invoke({"input":"Papers authoured by Jaeyeong Ryu"}))