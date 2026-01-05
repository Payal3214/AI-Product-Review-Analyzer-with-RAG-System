import streamlit as st
from rag import load_vectordb, fast_retriever

st.title("Import Test")

st.write(load_vectordb())
st.write(fast_retriever(None))
