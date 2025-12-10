# AI-Product-Review-Analyzer-with-RAG-System
This project is a Retrieval-Augmented Generation (RAG) system built to analyze Amazon shoe product reviews. The system allows users to query thousands of real customer reviews and receive structured, summarized, and sentiment-aware insights.


## Features
- Load and clean **Amazon shoe review dataset** (~85k reviews)  
- Automatic sentiment classification (`positive`, `neutral`, `negative`)  
- Frequent issues extraction using NLP  
- Retrieval-Augmented Generation pipeline for answering queries on product reviews  
- Word cloud visualization for common keywords per sentiment  
- Fully Python-based pipeline; ready for deployment



## Tech Stack
- **Python:** Pandas, NumPy, Regex  
- **NLP / AI:** HuggingFace Transformers, LangChain, SentenceTransformers  
- **Vector Database:** ChromaDB  
- **Visualization:** Matplotlib, WordCloud  
- **Notebook:** Jupyter  
- **Deployment (optional):** Streamlit



## Project Structure

amazon-shoes-rag/
│── data/
│ └── amazon_shoes_reviews.csv
│── notebooks/
│ └── amazon_shoes_rag.ipynb
│── requirements.txt
│── README.md
└── .gitignore


## How it Works
1. **Load Dataset** – Import Amazon shoe reviews CSV.  
2. **Clean Data** – Remove missing or non-English reviews.  
3. **Sentiment Analysis** – Label reviews as positive, neutral, or negative.  
4. **Frequent Issues Extraction** – Identify common complaints using NLP tokenization.  
5. **RAG Pipeline** – Convert reviews into embeddings and store in ChromaDB.  
6. **Query System** – Ask natural language questions, get structured answers summarizing multiple reviews.  

---

##  Example Queries
- "What are the most common issues customers have reported?"  
- "Summarize key strengths of this shoe product."  
- "Which features do customers like the most?"  

---

## 📈 Outcome
- Extracts insights from 85k+ reviews efficiently  
- Provides structured answers for product managers and business teams  
- Demonstrates AI-powered business intelligence for e-commerce  














