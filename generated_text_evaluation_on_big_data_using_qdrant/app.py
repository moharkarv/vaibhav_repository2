import streamlit as st
from retriever import load_resources, retrieve_relevant_text
from evaluation import evaluate_answer
from web_search import FactCheckerAgent
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from traditional_evaluation import evaluate_generated_text
from retriever import load_resources
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest, SearchParams



# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERP_API_KEY = os.getenv("SERP_API_KEY", "")

st.set_page_config(page_title="LLM Evaluation UI", layout="wide")
st.title("🧠 LLM Evaluation Tool")


# Add this at the top of the script, after st.title(...)
st.sidebar.header("⚙️ Options")
enable_web_search  = st.sidebar.checkbox("🌐 Enable Web Search", value=False)




client = QdrantClient(
    url="ht", 
    api_key="enter key",
)


# 🔒 Cached loading of resources (model, FAISS, BM25, embeddings)
@st.cache_resource(show_spinner="Loading resources...")
def get_cached_resources():
    return load_resources()

# try:
#     model, combined_embeddings, combined_metadata, bm25 = get_cached_resources()
# except Exception as e:
#     st.error(f"⚠️ Resource loading failed: {e}")
#     st.stop()

try:
    model, bm25 = get_cached_resources()
except Exception as e:
    st.error(f"⚠️ Resource loading failed: {e}")
    st.stop()






# 🔍 Input Section
st.header("🔍 Enter Inputs for Evaluation")

user_query = st.text_area("1️⃣ User Query", placeholder="What is the capital of France?")
context = st.text_area("2️⃣ Context (from retrieval or knowledge base)", height=200, placeholder="France is a country in Europe. Its capital is Paris...")
generated_text = st.text_area("3️⃣ Generated Answer", height=200, placeholder="The capital of France is Paris.")
#reference_input= st.text_area("3️⃣ Reference Input", height=200, placeholder="Guideline related to text")
reference_input = st.text_area("4️⃣ Reference Input", height=200, placeholder="Guideline related to text")

# 🚀 Retrieval & Display only when Submit is clicked
if st.button("✅ Submit"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a User Query.")
    else:
        with st.spinner("🔍 Retrieving relevant content..."):
            # cleaned_text = retrieve_relevant_text(
            #     query=generated_text,
            #     model=model,
            #     index=index,
            #     combined_embeddings=combined_embeddings,
            #     combined_metadata=combined_metadata,
            #     bm25=bm25
            # )

        #     cleaned_text = retrieve_relevant_text(
        #     query=generated_text,
        #     model=model,
        #     client=client,
        #     collection_name="my_embeddings",
        #     bm25=bm25
        # )  
###########################################################################
            #best
        #     cleaned_text = retrieve_relevant_text(
        #     query=generated_text,
        #     model=model,
        #     client=client,
        #     collection_name="my_embeddings",
        #     combined_embeddings=combined_embeddings,
        #     combined_metadata=combined_metadata,
        #     bm25=bm25
        # )
#########################################################################################
            cleaned_text = retrieve_relevant_text(
            query=generated_text,
            model=model,
            client=client,
            collection_name="my_embeddings",
            bm25=bm25,
            top_k=50,
            final_k=5,
            alpha=0.5
        )




            for i, doc in enumerate(cleaned_text, 1):
                st.markdown(f"<h4 style='color:#2B6CB0;'>Result {i}:</h4>", unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f9fafb;
                        border-left: 4px solid #3182ce;
                        padding: 12px 15px;
                        border-radius: 5px;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        font-size: 15px;
                        line-height: 1.5;
                        white-space: pre-wrap;
                        color: #1a202c;
                        box-shadow: 0 1px 3px rgb(0 0 0 / 0.1);
                        margin-bottom: 10px;
                    ">
                    {doc}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                    
#################################

            report=evaluate_answer(user_query,generated_text,cleaned_text,context,reference_input)


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

        st.success("✅ Retrieval complete!")
############################################################################################################################
        st.subheader("📋 Classic NLP Evaluation Summary")
        numbered_text = "\n\n".join([f"{i+1}. {item}" for i, item in enumerate(cleaned_text)])
        tool_check = evaluate_generated_text(
            generated_text=generated_text,
            context_docs=numbered_text
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



##############################################################################################################################      
        # 🎯 Display results
        #st.subheader("✏️ User Query")
        #st.write(user_query)

        #st.subheader("🔍 Retrieved Cleaned Text")
        #st.write(cleaned_text)

        #st.subheader("🤖 Generated Answer")
        #st.write(generated_text)

        #st.subheader("📚 Context")
        #st.write(context)


                
        # 🌐 Optional Web Fact-Check
        if enable_web_search and generated_text.strip():
            st.subheader("🌐 Web-Based Fact Check")
            st.write("Checking factuality using live internet search...")

            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.4,
                    google_api_key=GEMINI_API_KEY
                )
                agent = FactCheckerAgent(
                    llm=gemini_llm,
                    gemini_api_key=GEMINI_API_KEY,
                    serp_api_key=SERP_API_KEY
                )

                fact_result = agent.run_fact_check(generated_text, generated_text)

                st.success("✅ Web Fact-Check Result:")
                st.markdown("#### ✅ Fact-Check Summary:")
                st.markdown(fact_result.strip())

            except Exception as e:
                st.error(f"❌ Web fact-check failed: {e}")

