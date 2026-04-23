import os
import shutil
import tempfile
from typing import List

import chromadb
import pypdf
import streamlit as st
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "policy_chroma_db")

COLLECTION_NAME = "policy_docs"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "BAAI/bge-small-en"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
SIMILARITY_TOP_K = 5


# =========================
# Global model settings
# =========================
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# =========================
# Chroma helpers
# =========================
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_chroma_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(COLLECTION_NAME)


def reset_chroma_collection():
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION_NAME)


# =========================
# File extraction helpers
# =========================
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        return "\n\n".join(pages).strip()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    return ""


# =========================
# Indexing helpers
# =========================
def build_index_from_uploaded_files(uploaded_files) -> VectorStoreIndex:
    """
    Save uploaded files temporarily, extract text, chunk it, and index it in Chroma.
    """
    collection = reset_chroma_collection()

    with tempfile.TemporaryDirectory() as tmpdir:
        documents: List[Document] = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmpdir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            extracted_text = extract_text_from_file(file_path)

            if extracted_text and extracted_text.strip():
                documents.append(
                    Document(
                        text=extracted_text,
                        metadata={"file_name": uploaded_file.name},
                    )
                )

        if not documents:
            raise ValueError(
                "No readable text was extracted from the uploaded files."
            )

        first_text = documents[0].text.lstrip()
        if first_text.startswith("%PDF-"):
            raise ValueError(
                "The uploaded PDF text was not extracted correctly. "
                "Try another PDF or convert it to a TXT file first."
            )

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[Settings.node_parser],
            embed_model=Settings.embed_model,
        )

        chunk_count = collection.count()

        st.session_state.index_debug = {
            "document_count": len(documents),
            "chunk_count": chunk_count,
        }

        if chunk_count == 0:
            raise ValueError(
                "Indexing finished but no chunks were stored. "
                "The extracted text may be empty or the PDF may not be text-friendly."
            )

        return index


def load_existing_index() -> VectorStoreIndex:
    """
    Load an already-built Chroma collection for querying.
    """
    collection = get_chroma_collection()
    chunk_count = collection.count()

    if chunk_count == 0:
        raise ValueError("The policy database is empty. Index files first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    st.session_state.index_debug = {
        "document_count": st.session_state.index_debug.get("document_count", 0),
        "chunk_count": chunk_count,
    }

    return index


# =========================
# Query / generation helpers
# =========================
def build_policy_query_engine(index: VectorStoreIndex):
    qa_prompt = PromptTemplate(
        "You are an internal policy and compliance assistant. "
        "Answer the question using only the policy context provided. "
        "If the answer is not clearly supported by the policy text, say that explicitly.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n\n"
        "Return your answer in this format:\n"
        "Answer: <clear grounded answer>\n"
        "Citation: <policy wording, section name, or quoted evidence>\n"
        "Compliance Note: <brief risk, caution, or follow-up note if relevant>"
    )

    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt}
    )
    return query_engine


def extract_text_from_response(response) -> str:
    if hasattr(response, "response") and response.response:
        text = response.response.strip()
        if text:
            return text

    text = str(response).strip()
    if text and text != "Empty Response":
        return text

    return (
        "No answer was generated. This usually means the model returned nothing "
        "or the indexed content was not retrieved properly."
    )


def generate_summary(index: VectorStoreIndex) -> str:
    summary_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    prompt = (
        "Create an employee-friendly summary of the uploaded policy documents. "
        "Include these sections:\n"
        "1. One-page Summary\n"
        "2. Bullet-Point Highlights\n"
        "3. What Employees Need to Know\n"
        "4. Key Compliance Risks or Common Mistakes\n"
        "Use only information grounded in the uploaded policies."
    )

    response = summary_engine.query(prompt)
    return extract_text_from_response(response)


def generate_quiz(index: VectorStoreIndex) -> str:
    quiz_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    prompt = (
         "Create a training quiz based only on the uploaded policy documents.\n\n"

        "Format EXACTLY like this:\n\n"

        "Question 1: <question>\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option>\n\n"
    
        "Correct Answer: <letter>\n"
        "Explanation: <one sentence>\n\n"

        "Repeat this format for 5 questions.\n\n"

        "Each answer choice MUST be on its own line.\n"
        "Do NOT put multiple answer choices on the same line."
    )
    response = quiz_engine.query(prompt)
    return extract_text_from_response(response)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="AI Policy & Compliance Assistant", layout="wide")

st.title("AI-Enabled Internal Policy & Compliance Assistant")
st.caption(
    "Upload policy PDFs or TXT files, ask grounded questions, generate summaries and quizzes, "
    "and review unclear language."
)

with st.sidebar:
    st.header("System Settings")
    st.write(f"**LLM:** {OLLAMA_MODEL}")
    st.write(f"**Embedding Model:** {EMBED_MODEL}")
    st.write(f"**Chunk Size:** {CHUNK_SIZE}")
    st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
    st.write(f"**Top K Retrieval:** {SIMILARITY_TOP_K}")
    st.write("**Accepted upload types:** PDF, TXT")


    
if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "index_debug" not in st.session_state:
    st.session_state.index_debug = {}

uploaded_files = st.file_uploader(
    "Upload policy files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Index Uploaded Policies", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Indexing policies..."):
                try:
                    build_index_from_uploaded_files(uploaded_files)
                    st.session_state.docs_indexed = True
                    st.success("Policies indexed successfully.")
                except Exception as e:
                    st.session_state.docs_indexed = False
                    st.error(f"Indexing failed: {e}")

with col2:
    if st.button("Load Existing Policy Database", use_container_width=True):
        with st.spinner("Loading policy database..."):
            try:
                load_existing_index()
                st.session_state.docs_indexed = True
                st.success("Existing policy database loaded.")
            except Exception as e:
                st.session_state.docs_indexed = False
                st.error(f"Could not load database: {e}")

st.markdown("---")
st.subheader("Policy Q&A")

user_question = st.text_input(
    "Ask a policy question",
    placeholder="Example: Can I use my personal laptop for work?",
)

if st.button("Get Answer"):
    if not st.session_state.docs_indexed:
        st.warning("Please index uploaded policies or load an existing policy database first.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                index = load_existing_index()
                query_engine = build_policy_query_engine(index)
                response = query_engine.query(user_question)
                answer = extract_text_from_response(response)

                st.session_state.qa_history.append(
                    {"question": user_question, "answer": answer}
                )

                st.markdown("### Answer")
                st.write(answer)


            except Exception as e:
                st.error(f"Q&A failed: {e}")

st.markdown("---")
st.subheader("Policy Actions")

a1, a2 = st.columns(2)

with a1:
    if st.button("Generate Summary", use_container_width=True):
        if not st.session_state.docs_indexed:
            st.warning("Please index or load policy documents first.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    index = load_existing_index()
                    summary = generate_summary(index)
                    st.markdown("### Policy Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

with a2:
    if st.button("Generate Quiz", use_container_width=True):
        if not st.session_state.docs_indexed:
            st.warning("Please index or load policy documents first.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    index = load_existing_index()
                    quiz = generate_quiz(index)
                    st.markdown("### Training Quiz")
                    st.text(quiz)
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")

if st.session_state.index_debug:
    st.markdown("---")
    st.subheader("Index Diagnostics")
    st.write(f"Documents loaded: {st.session_state.index_debug.get('document_count', 0)}")
    st.write(f"Chunks indexed: {st.session_state.index_debug.get('chunk_count', 0)}")

if st.session_state.qa_history:
    st.markdown("---")
    st.subheader("Previous Questions")
    for item in reversed(st.session_state.qa_history):
        with st.expander(item["question"]):
            st.write(item["answer"])
