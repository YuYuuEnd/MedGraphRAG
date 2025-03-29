from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from data_loader import data_load
from utils import *

class NaiveRAG:
    retriever_global = None

    def Indexing(self, file_paths, chunk_size = 500, chunk_overlap = 20):
        global retriever_global
        data = []
        for file_path in file_paths:
            data += data_load(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=data, embedding=embedding)
        print("DONE Index!")

        retriever_global = vectorstore.as_retriever()

    def Retrieve(self, query, k=5):
        if retriever_global is None:
            raise ValueError("No indexing data")

        retriever_global.search_type = "mmr"
        # retriever_global.search_kwargs = {"lambda_mult": 0.5}  # Adjust lambda_mult as needed

        relevant_docs = retriever_global.invoke(query)[:k]
        # print(relevant_docs)

        print("DONE Retrieve!")
        return relevant_docs
    
    def Generate(self, sys, user):
        related_doc = self.Retrieve(user, k=5)
        docs = "\n\n".join(doc.page_content for doc in related_doc)

        user = f"""
        Generate answer for user question based on documents provided.
        Documents:
        {docs}

        Question:
        {user}

        Answer:
        """

        # print(user)
        response = call_llm(sys, user)
        return response
