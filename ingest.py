import os
from scraping import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from qdrant_client import QdrantClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import QdrantVectorStore

load_dotenv()

class DocumentIngestionPipeline:
    def __init__(self, 
                 links_file: str, 
                 embedding_model_name: str, 
                 qdrant_url: str, 
                 qdrant_api_key: str, 
                 collection_name: str, 
                 chunk_size: int = 256, 
                 chunk_overlap: int = 100):
        self.links_file = links_file
        self.embedding_model_name = embedding_model_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"trust_remote_code": True, "device": "cuda"}
        )

        self.namespace = f"qdrant/{self.collection_name}"
        self.db_file = f"{self.collection_name}.sql"

    def load_links(self):
        with open(self.links_file, "r", encoding='utf-8') as f:
            links = f.read().splitlines()
        return links

    def load_documents(self):
        links = self.load_links()
        loader = SeleniumURLLoader(links)
        return loader.load()

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)

    def initialize_qdrant(self):
        if not self.qdrant_client.collection.exists(self.collection_name):
            QdrantVectorStore.from_texts(
                ["This is a test text"],
                self.embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name
            )

    def initialize_record_manager(self):
        record_manager = SQLRecordManager(namespace=self.namespace, db_url=f"sqlite:///{self.db_file}")
        if not os.path.exists(self.db_file):
            record_manager.create_schema()
        return record_manager

    def run(self):
        documents = self.load_documents()
        split_docs = self.split_documents(documents)
        self.initialize_qdrant()
        record_manager = self.initialize_record_manager()
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        index(split_docs, record_manager, vector_store, cleanup="full", source_id_key="source")


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline(
        links_file="links.txt",
        embedding_model_name="BAAI/bge-m3",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="isfcr_bge_m3_256_100"
    )
    pipeline.run()