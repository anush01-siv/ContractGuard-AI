import os
import io
from typing import List, Dict, Any
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader
import docx

st.set_page_config(page_title="ContractGuard AI", page_icon="🛡️", layout="wide")

MASTER_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not MASTER_GROQ_API_KEY:
    st.error("⚠️ Groq API key not found. Please set 'GROQ_API_KEY' in Streamlit secrets or environment variables.")
    st.stop()

os.environ["GROQ_API_KEY"] = MASTER_GROQ_API_KEY


DEFAULT_CONTRACT_CHUNKS = [
    {
        "id": "doc_1",
        "section": "Section 4.2 - Liability Caps",
        "page": 12,
        "text": "The total aggregate liability of the Contractor under this Master Services Agreement (MSA) shall not exceed $1,000,000 USD or the total fees paid in the preceding 12 months, whichever is lower."
    },
    {
        "id": "doc_2",
        "section": "Section 9.1 - Termination for Convenience",
        "page": 24,
        "text": "Either party may terminate this agreement without cause by providing at least sixty (60) days written notice to the other party. Failure to provide notice incurs a $50,000 penalty."
    },
    {
        "id": "doc_3",
        "section": "Section 11.4 - Data Privacy & GDPR",
        "page": 31,
        "text": "Data Processor shall notify Data Controller within twenty-four (24) hours of becoming aware of any Personal Data Breach impacting EU citizens, complying with GDPR Article 33."
    }
]

def parse_pdf(file) -> List[Dict[str, Any]]:
    reader = PdfReader(file)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            # Chunking text into ~500 character blocks for precision
            sub_chunks = [text[j:j+500] for j in range(0, len(text), 450)]
            for sub_idx, chunk_text in enumerate(sub_chunks):
                chunks.append({
                    "id": f"pdf_p{i+1}_{sub_idx}",
                    "section": f"Page {i+1} - Segment {sub_idx+1}",
                    "page": i + 1,
                    "text": chunk_text.strip()
                })
    return chunks

def parse_docx(file) -> List[Dict[str, Any]]:
    doc = docx.Document(file)
    chunks = []
    current_section = "General Terms"
    
    for idx, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue
        # Treat headings as section markers
        if p.style.name.startswith('Heading'):
            current_section = text
        else:
            chunks.append({
                "id": f"docx_p{idx}",
                "section": current_section,
                "page": 1,
                "text": text
            })
    return chunks

def parse_txt(file) -> List[Dict[str, Any]]:
    text = file.read().decode("utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for idx, p in enumerate(paragraphs):
        chunks.append({
            "id": f"txt_{idx}",
            "section": f"Paragraph {idx+1}",
            "page": 1,
            "text": p
        })
    return chunks


@st.cache_resource(show_spinner=False)
def load_dense_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

class ContractGuardEngine:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        
        # BM25 Sparse Index
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # FAISS Dense Index
        self.encoder = load_dense_encoder()
        embeddings = self.encoder.encode(self.texts, convert_to_numpy=True)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def bm25_search(self, query: str, top_k: int = 4) -> List[int]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return list(np.argsort(scores)[::-1][:top_k])

    def dense_search(self, query: str, top_k: int = 4) -> List[int]:
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        _, indices = self.index.search(query_vec, top_k)
        return list(indices[0])

    def reciprocal_rank_fusion(self, bm25_ranks: List[int], dense_ranks: List[int], k: int = 60) -> List[int]:
        rrf_scores = {}
        for rank, doc_idx in enumerate(bm25_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))
            
        return sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        bm25_hits = self.bm25_search(query, top_k=4)
        dense_hits = self.dense_search(query, top_k=4)
        fused_indices = self.reciprocal_rank_fusion(bm25_hits, dense_hits)[:top_k]
        return [self.chunks[idx] for idx in fused_indices]

def audit_contract(query: str, engine: ContractGuardEngine) -> tuple[str, List[Dict[str, Any]]]:
    retrieved_chunks = engine.retrieve(query, top_k=2)
    
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[{chunk['section']} | Page {chunk['page']}]\n\"{chunk['text']}\""
        )
    context_str = "\n\n".join(context_blocks)
    
    prompt_template = """You are an Enterprise Legal AI Auditor. Answer the user question based strictly on the provided contract context below.

Rule 1: You must explicitly cite the Section/Page number for every factual claim or assertion.
Rule 2: Include verbatim quote snippets to justify your audit conclusion.
Rule 3: If the context does not contain sufficient details to answer the query, explicitly state 'Insufficient Contract Evidence'.

Contract Context:
{context}

Question: {query}

Legal Audit Report:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
        groq_api_key=MASTER_GROQ_API_KEY
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "query": query})
    return response.content, retrieved_chunks


st.title("🛡️ ContractGuard AI — Enterprise Legal Auditor")
st.caption("Hybrid BM25 + FAISS Vector Search Powered by Groq `openai/gpt-oss-20b`")

# Sidebar Upload Controls
st.sidebar.header("📂 Document Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload a Contract (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    with st.spinner(f"Parsing {uploaded_file.name}..."):
        if file_ext == "pdf":
            active_chunks = parse_pdf(uploaded_file)
        elif file_ext == "docx":
            active_chunks = parse_docx(uploaded_file)
        elif file_ext == "txt":
            active_chunks = parse_txt(uploaded_file)
        st.sidebar.success(f"Loaded {len(active_chunks)} document chunks!")
else:
    st.sidebar.info("Using sample fallback contract. Upload your own file above!")
    active_chunks = DEFAULT_CONTRACT_CHUNKS


engine = ContractGuardEngine(active_chunks)

st.markdown("### 🔍 Audit Query")
user_query = st.text_input(
    "Enter a legal audit question:",
    value="What is our maximum aggregate liability limit and what happens if we terminate early without cause?"
)

if st.button("Run Legal Audit", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Executing Hybrid RRF Retrieval & Audit Synthesis..."):
            audit_report, sources = audit_contract(user_query, engine)
            
            st.markdown("---")
            st.markdown("### 📊 Legal Audit Report")
            st.success(audit_report)
            
            st.markdown("### 📌 Retracted Source Evidence")
            for idx, source in enumerate(sources, 1):
                with st.expander(f"Source {idx}: {source['section']} (Page {source['page']})"):
                    st.write(f"**Text:** \"{source['text']}\"")
                    st.write(f"**Chunk ID:** `{source['id']}`")