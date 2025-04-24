from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import argparse
CHROMA_PATH = "chroma"


#loading the pdfs
def load_documents():
    document_loader = PyPDFDirectoryLoader('pdfs')
    return document_loader.load()
# documents = load_documents()
# print(documents[0])

#split the documents
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
documents = load_documents()
chunks = split_documents(documents)
# print(chunks[0])

#embedding function
def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#creating a database
def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embeddings_function()
    )

    #calculate page ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    #add or update the documents
    existing_items = db.get(include=[])   #IDs are always included by default
    existing_ids =set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    #only add documents that don't exist in the DB
    new_chunks=[]
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks,ids=new_chunks_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    #this will create IDs like "pdfs/pdf_name:6:2"
    """
    args
    what the function does
    return"""
    #page source:page number: chunk index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id= f"{source}:{page}"

        #if the page ID is the same as the last one, increament the index
        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index = 0

        #calculate the chunk ID
        chunk_id=f"{current_page_id}:{current_chunk_index}"
        last_page_id=current_page_id

        #add it to the page meta-data
        chunk.metadata["id"] = chunk_id

    return chunks

PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    
    Answer the question based on the above context: {question}
    """

def query_rag(query_text: str):
    embedding_function=get_embeddings_function()
    db=Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    print(db)
    results = db.similarity_search_with_score(query_text,k=10)
    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])
    with open("cont.txt","w") as f:
        f.write(context_text)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(context_text)
    print(prompt)
    
    model = Ollama(model="mistral:instruct")
    response_text = model.invoke(prompt)
    print(response_text)
    
    sources =[doc.metadata.get("id",None) for doc,_score in results]
    formatted_response = f"Response: {response_text}\nSources:{sources}"
    print(formatted_response)
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str,help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    query_rag(query_text)

if __name__ == "__main__":
    main()

