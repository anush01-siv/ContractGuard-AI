# 🛡️ ContractGuard AI — Enterprise Hybrid RAG Engine

A production-ready, security-first Hybrid Retrieval-Augmented Generation (RAG) platform engineered for legal contract auditing, compliance verification, and granular clause extraction.

## ⚡ Key Architecture & Features

- **Hybrid Search Retrieval (BM25 + FAISS):** Combines lexical keyword search (`rank_bm25`) with dense vector embeddings (`FAISS` + `sentence-transformers`) to guarantee high clause recall across complex legal documentation.
- **Reciprocal Rank Fusion (RRF):** Merges keyword and vector scores to surface exact matches alongside contextual semantic agreement.
- **Page-Level Citation Tracing:** Tracks exact source segments and page numbers for transparent compliance auditing.
- **Zero-Retention Ephemeral Security:** Operates in transient memory without persistent file retention, safeguarding confidential enterprise documents.

---

## 🛠️ Tech Stack

- **UI Framework:** Streamlit
- **LLM Orchestration:** LangChain, Groq API (Llama 3)
- **Vector Indexing & Embeddings:** FAISS, `sentence-transformers/all-MiniLM-L6-v2`
- **Lexical Retrieval:** `rank_bm25` (BM25Okapi)
- **Document Parsing:** PyPDF, `python-docx`

---

## 📜 Enterprise Security & Privacy

- API keys are handled strictly through standard environment variables and `st.secrets`.
- Document processing runs transiently per session; uploaded artifacts are automatically scrubbed on session termination.
