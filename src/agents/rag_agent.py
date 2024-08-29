import os 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RAGAgent: 
    def __init__(self, doc_path, embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")):
        self.embedding_model = embedding_model
        self.doc_path = doc_path

    def load_data(self):
        doc_files = [f for f in os.listdir(self.doc_path) if f.endswith(".md") and os.path.isfile(os.path.join(self.doc_path, f))]
        document_data = []
        for file_name in doc_files: 
            loader = TextLoader(os.path.join(self.doc_path, file_name))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            docs = text_splitter.split_documents(documents)
            document_data = document_data + docs

        RAG_db = Chroma.from_documents(document_data, self.embedding_model)

        return RAG_db.as_retriever(search_type="similarity")


    def create_RAG_tool(self): 
        name = "RAG Tool"
        description = "Search and retrieve data from the documents provided by the user."

        return create_retriever_tool(self.load_data(), name, description)

