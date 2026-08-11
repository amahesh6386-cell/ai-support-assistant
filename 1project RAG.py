import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# Environment variables load cheyadam
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Gemini Client Initialize
client = genai.Client(api_key=api_key)

# Local Embedding Model (Free - Runs on CPU)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# Helper Functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# Streamlit UI Setup
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚")
st.title("📚 Custom PDF Knowledge Base (RAG System)")
st.write("Upload a PDF document and ask any questions based on its content!")

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    if "processed" not in st.session_state:
        with st.spinner("Extracting text and building vector database..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(raw_text)
            index, embeddings = build_vector_store(chunks)
            
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed = True
            st.success("PDF Processed Successfully!")

    st.subheader("❓ Ask a Question")
    user_query = st.text_input("Enter your question about the PDF:")

    if user_query:
        with st.spinner("Searching document & generating answer..."):
            query_vector = embedder.encode([user_query]).astype("float32")
            k = 3
            distances, indices = st.session_state.index.search(query_vector, k)
            
            retrieved_chunks = [st.session_state.chunks[i] for i in indices[0] if i < len(st.session_state.chunks)]
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are a helpful assistant. Answer the user's question ONLY using the provided Context below.
If the answer is not contained in the Context, say "I cannot find the answer in the uploaded document."

Context:
{context}

Question:
{user_query}

Answer:
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            st.markdown("### 💡 Answer:")
            st.write(response.text)

            with st.expander("🔍 View Retrieved Context Chunks (Source Data)"):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.info(chunk)