import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectordb():
    persist_dir = "models/chroma_shoes"
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embed)
    else:
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(persist_directory=persist_dir, embedding_function=embed)

