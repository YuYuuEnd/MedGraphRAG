from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json

def data_load(pdf_file_path, chunk_size=500, chunk_overlap=20):
    # Load and initial chunks from pdf file
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    chunks = splitter.split_documents(documents)
    return chunks

def load_from_json(json_file_path):
    # Load nodes from json file
    with open(json_file_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    return nodes

def load_to_json(json_file_path, nodes):
    # Load nodes to json file
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)
