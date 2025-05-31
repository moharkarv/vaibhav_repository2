# rag_pipeline.py
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# Load Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def chunk_by_hierarchy(elements):
    chunks = []
    chapter = None
    section = None
    paragraph_accumulator = []

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        
        if element.category == "Title":
            if paragraph_accumulator:
                chunks.append({
                    "chapter": chapter,
                    "section": section,
                    "paragraph": " ".join(paragraph_accumulator)
                })
                paragraph_accumulator = []

            chapter = text
            section = None

        elif element.category == "Header":
            if paragraph_accumulator:
                chunks.append({
                    "chapter": chapter,
                    "section": section,
                    "paragraph": " ".join(paragraph_accumulator)
                })
                paragraph_accumulator = []

            section = text

        elif element.category == "NarrativeText":
            paragraph_accumulator.append(text)

    if paragraph_accumulator:          #checks if the paragraph_accumulator list is not empty.
        chunks.append({
            "chapter": chapter,
            "section": section,
            "paragraph": " ".join(paragraph_accumulator)
        })

    return chunks

def process_pdf(pdf_path, embeddings):
    pdf_elements = partition_pdf(filename=pdf_path, languages=['eng'], strategy="fast")
    chunked_elements = chunk_by_hierarchy(pdf_elements)

    documents = []
    for element in chunked_elements:
        metadata = {
            "chapter": element.get("chapter", None),
            "section": element.get("section", None)
        }
        doc = Document(page_content=element["paragraph"].lower().strip(), metadata=metadata)
        documents.append(doc)

    # Vector index
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever_vectordb = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )


#     Parameter	Meaning
# fetch_k=10	First, retrieve the top 10 most relevant documents from the vector DB (based on cosine similarity to the query)
# k=4	Then, use MMR to choose 4 diverse and relevant documents from those 10

    bm25_retriever = BM25Retriever.from_documents(documents)

    def hybrid_retrieval(query):
        bm25_results = bm25_retriever.get_relevant_documents(query)[:3]  
        vector_results = retriever_vectordb.get_relevant_documents(query)[:3]
        combined = {doc.page_content: doc for doc in (bm25_results + vector_results)}
        return list(combined.values())

    return hybrid_retrieval
