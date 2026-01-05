from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

PERSIST_DIR = "chroma_db"  # your existing persist directory

def load_vectordb(force_recreate=False):
    """
    Load or create the Chroma vector store.
    If the embedding function conflicts with an existing collection,
    optionally delete and recreate it.
    
    Args:
        force_recreate (bool): If True, deletes the existing DB and starts fresh.
    """
    embed = OpenAIEmbeddings()  # your embedding function

    # If you want to force reset or the directory is corrupted
    if force_recreate and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"[INFO] Deleted existing Chroma DB at {PERSIST_DIR}")

    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed)
        print("[INFO] Loaded existing Chroma vector store")
    except ValueError as e:
        # This handles embedding function mismatch
        print("[WARNING] Embedding function conflict detected. Recreating DB...")
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed)
        print("[INFO] Created new Chroma vector store with current embedding function")

    return vectordb


