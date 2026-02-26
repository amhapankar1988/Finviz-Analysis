import os
import time
import requests
import pandas as pd
import streamlit as st
from finvizfinance.screener.overview import Overview
from finvizfinance.quote import finvizfinance as FinvizQuote
from huggingface_hub import HfApi, snapshot_download
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Finviz AI Terminal", layout="wide")

# --- SECRETS & REPO ---
HF_TOKEN = os.getenv("HF_TOKEN")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
# Replace with your actual dataset repo ID (e.g., "username/trading-strategy-db")
DATASET_REPO_ID = "amhapankar/my-trading-brain" 

# --- INITIALIZE AI ---
@st.cache_resource
def init_ai():
    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://ca-tor.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
        apikey=WATSONX_APIKEY,
        params={"decoding_method": "sample", "max_new_tokens": 800, "temperature": 0.2}
    )
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeds

llm, embeddings = init_ai()

# Initialize session state for the vector DB
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# --- PERSISTENCE LOGIC ---
def save_to_hf_dataset():
    """Saves the FAISS index locally and uploads it to HF Datasets."""
    if st.session_state.vector_db:
        folder_name = "faiss_index_fv"
        st.session_state.vector_db.save_local(folder_name)
        
        api = HfApi()
        try:
            api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=folder_name,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            st.sidebar.success("☁️ Brain Synced to Dataset!")
        except Exception as e:
            st.sidebar.error(f"Sync Error: {e}")

def load_from_hf_dataset():
    """Downloads the FAISS index from HF Datasets and loads it into the app."""
    try:
        # Download the entire folder from the dataset repo
        local_path = snapshot_download(
            repo_id=DATASET_REPO_ID, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        st.session_state.vector_db = FAISS.load_local(
            local_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return True
    except Exception:
        return False

# --- UI SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    
    # Try to auto-load on startup
    if st.session_state.vector_db is None:
        if load_from_hf_dataset():
            st.success("✅ Knowledge Base Loaded from Cloud")
        else:
            st.info("💡 No cloud brain found. Upload PDFs to start.")

    st.divider()
    files = st.file_uploader("Upload Strategy PDFs", accept_multiple_files=True)
    if st.button("🧠 Train & Sync to Cloud") and files:
        with st.spinner("Indexing and Uploading..."):
            docs = []
            for f in files:
                with open(f.name, "wb") as tmp: tmp.write(f.read())
                docs.extend(PyPDFLoader(f.name).load())
            
            splits = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)
            st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
            save_to_hf_dataset()

# --- MAIN TERMINAL ---
st.title("📉 Finviz AI Strategy Terminal")

if st.button("🔥 START STRATEGIC SCAN"):
    if not st.session_state.vector_db:
        st.error("Knowledge base is empty. Please upload strategy books in the sidebar.")
    else:
        with st.spinner("Running Triple Screen..."):
            f_screen = Overview()
            f_screen.set_filter(filters_dict={
                'EPS growthqtr over qtr': 'Over 25%',
                'Relative Volume': 'Over 1.5',
                'Price': 'Over $20',
                '200-Day Simple Moving Average': 'Price above SMA200'
            })
            df = f_screen.screener_view(order='Relative Volume', ascend=False)
            
            if df is not None and not df.empty:
                candidates = df['Ticker'].tolist()[:3]
                
                for ticker in candidates:
                    # Technicals
                    stock = FinvizQuote(ticker)
                    fundament = stock.ticker_fundament()
                    price = fundament.get('Price', 'N/A')
                    news_df = stock.ticker_news()
                    top_news = news_df['Title'].tolist()[:3] if not news_df.empty else ["No news found"]
                    
                    # Refined RAG Search
                    docs = st.session_state.vector_db.similarity_search(f"buy rules pivot entry for {ticker}", k=3)
                    context = "\n".join([d.page_content for d in docs])
                    
                    # Analysis
                    final_prompt = f"Strategy: {context}\nData: {ticker} at {price}. News: {top_news}. verdict?"
                    result = llm.invoke(final_prompt)
                    
                    with st.expander(f"📊 {ticker} Analysis", expanded=True):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.image(f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l")
                        with c2:
                            st.write(result.content)
            else:
                st.info("No candidates found meeting the 'Wizard' criteria.")