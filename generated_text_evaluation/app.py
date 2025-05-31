import streamlit as st
import pandas as pd
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import os

from PyPDF2 import PdfReader
from rag_pipeline import process_pdf
from evaluation import evaluate_answer
from web_search import FactCheckerAgent
from url_data import extract_text_from_url
from hybrid_search import hybrid_search,prepare_index_from_csv  # Ensure you import this
from traditional_evaluation import evaluate_generated_text


# Load embedding model
csvmodel = SentenceTransformer("all-MiniLM-L6-v2")

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "enter_key")
SERP_API_KEY = os.getenv("SERP_API_KEY", "enter_key")

# --- Streamlit Setup ---
st.set_page_config(page_title="Multi-Input Evaluation App", layout="centered")
st.title("📥 Generated Text Evaluation")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    enable_web_search = st.toggle("Enable Web Search for Fact-Check", value=False)

# --- Input Selection ---
file_type = st.radio("Select input type:", ["PDF", "CSV", "Plain Text", "URL"])
input_text = ""
uploaded_file = None
df, index, bm25 = None, None, None
rag_results = []


@st.cache_resource(show_spinner="🔄 Indexing CSV file...")
def cached_prepare_index(csv_path):
    return prepare_index_from_csv(csv_path)


# --- File Upload Handling ---
if file_type in ["PDF", "CSV"]:
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "csv"])
    if uploaded_file and file_type == "CSV":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            csv_path = tmp_file.name

        df, index, bm25 = prepare_index_from_csv(csv_path)
        st.dataframe(df)

elif file_type == "Plain Text":
    input_text = st.text_area("📄 Enter your plain text here:", height=200)

elif file_type == "URL":
    url_input = st.text_input("Enter the URL here:")
    if url_input:
        try:
            extracted_text = extract_text_from_url(url_input)
            input_text = extracted_text
            st.text_area("Extracted Text from URL", value=input_text, height=300)
        except Exception as e:
            st.error(f"Failed to extract text from URL: {e}")

# --- Manual Text Inputs ---
st.subheader("🧾 Evaluation Inputs")
generated_text = st.text_area("Generated Text", height=150)
context = st.text_area("Context", height=100)
reference = st.text_area("Reference (Optional)", height=100)
user_query = st.text_area("User Query For Evaluation Model", height=100)

@st.cache_resource
def load_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


@st.cache_resource(show_spinner="🔍 Processing PDF...")
def cached_process_pdf(pdf_path,_model):
    return process_pdf(pdf_path,_model)




# --- Submit Button Logic ---
if st.button("Submit"):
    st.subheader("📊 Results")
    context_docs = ""

    if file_type == "PDF" and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.write("🔍 Extracting contexts from PDF...")
        hybrid_retriever = cached_process_pdf(pdf_path,model)
        rag_results = hybrid_retriever(generated_text)
        st.success("✅ Retrieved Contexts using Hybrid RAG:")
        for i, doc in enumerate(rag_results, 1):
            st.markdown(f"**Result {i}:**")
            st.write(doc.page_content)
            if doc.metadata:
                st.caption(f"Metadata: {doc.metadata}")

                
    elif file_type == "CSV" and uploaded_file and df is not None:
        input_text = hybrid_search(generated_text, csvmodel, df, index, bm25)
        st.success("✅ Retrieved Context from CSV:")
        st.text_area("Hybrid Search Context", value=input_text, height=300)

    # Final fallback context
    context_docs = "\n\n".join([doc.page_content for doc in rag_results]) if rag_results else input_text
    
####################################################################################
        # --- Traditional Tool-Based Evaluation ---
    st.subheader("📋 Classic NLP Evaluation Summary")

    tool_check = evaluate_generated_text(
        generated_text=generated_text,
        context_docs=context_docs
    )

    st.markdown("#### 🧾 Grammar")
    st.write(f"**Score:** {tool_check['grammar']['score']} / 100")
    st.write(f"**Errors:** {tool_check['grammar']['errors']}")
    if tool_check['grammar']['suggestions']:
        st.markdown("**Suggestions:**")
        for suggestion in tool_check['grammar']['suggestions']:
            st.write(f"- {suggestion}")

    st.markdown("#### 📚 Factual Accuracy")
    st.write(f"**Confidence:** {tool_check['factual_accuracy_based_on_context']['confidence']}%")

    st.markdown("#### ✨ Style Metrics")
    for metric, val in tool_check["style"].items():
        st.write(f"**{metric.replace('_', ' ').title()}**: {val}")


###################################################################################################

    # --- Evaluation ---
    st.subheader("🧠 Quality Evaluation Report Using LLM")
    report = evaluate_answer(
        query=user_query,
        generated_text=generated_text,
        context_docs=context_docs,
        context_input=context,
        reference_input=reference,
    )

    if "error" in report:
        st.error(report["error"])
        st.text(report["raw_response"])
    else:
        st.metric("🧮 Composite Score", report["composite_score"])
        for section in ["grammar", "coherence", "factual_accuracy", "style"]:
            st.markdown(f"**{section.capitalize()}**: {report[section]['score']}/100")
            st.caption(report[section]['notes'])

        st.markdown("### 💡 Suggestions:")
        for suggestion in report["suggestions"]:
            st.write(f"- {suggestion}")

# --- Web Search (Optional) ---
if enable_web_search and generated_text:
    st.subheader("🌐 Web-Based Fact Check")
    st.write("Checking factuality using live internet search...")

    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, google_api_key=GEMINI_API_KEY)
    agent = FactCheckerAgent(
        llm=gemini_llm,
        gemini_api_key=GEMINI_API_KEY,
        serp_api_key=SERP_API_KEY
    )

    try:
        fact_result = agent.run_fact_check(generated_text, generated_text)
        st.success("✅ Web Fact-Check Result:")
        st.markdown("#### ✅ Fact-Check Summary:")
        st.markdown(fact_result.strip())
    except Exception as e:
        st.error(f"❌ Web fact-check failed: {e}")
