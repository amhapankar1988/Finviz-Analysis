---
title: Finviz AI Strategic Terminal
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered trading terminal using Finviz & RAG for strategic stock auditing.
---

# 🚀 Finviz AI Strategy Terminal

This Space hosts a specialized trading dashboard that combines **Finviz market scanning** with **RAG (Retrieval-Augmented Generation)**. It allows traders to upload their proprietary strategy books (PDFs) and have an AI model (Llama-3-3-70B) audit current market setups against those specific rules.

## 🛠️ Configuration Details

To run this Space, you must configure the following **Secrets** in your Space Settings:

1. **`WATSONX_APIKEY`**: Your IBM Watsonx.ai API Key.
2. **`WATSONX_PROJECT_ID`**: The Project ID from your Watsonx dashboard.
3. **`HF_TOKEN`**: A Hugging Face Access Token (Write access) to sync your Vector DB to your private dataset.

## 📂 Project Structure

* `app.py`: The main Streamlit application logic.
* `requirements.txt`: Python dependencies (Finvizfinance, LangChain, FAISS, etc.).
* `faiss_index_fv/`: Local storage for the vector database (synced to HF Datasets).

## ⚠️ Disclaimer
*This tool is for educational purposes only. Trading stocks involves significant risk. Always perform your own due diligence.*
