import pandas as pd
import numpy as np
import re
import faiss
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load embedding model once (globally)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Function 1: Process CSV and build indices
# -----------------------------------
def prepare_index_from_csv(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Merge columns into a single text field
    df['merge_text'] = df.apply(lambda row: ' '.join(f"{col}: {row[col]}," for col in df.columns), axis=1)

    # Clean text
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9.,!? ]+', '', text)
        return text.strip()

    df['clean_embedding_text'] = df['merge_text'].apply(clean_text)

    # Generate embeddings
    df['embedding'] = df['clean_embedding_text'].apply(lambda x: model.encode(x))

    # Create FAISS index
    embedding_matrix = np.vstack(df['embedding'].values)
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)

    # Create BM25 index
    df_tokens = df['clean_embedding_text'].apply(lambda x: x.lower().split()).tolist()
    bm25 = BM25Okapi(df_tokens)

    return df, index, bm25

# -----------------------------------
# Function 2: Perform Hybrid Search
# -----------------------------------
def hybrid_search(query, model, df, index, bm25, top_k=100, final_k=5, alpha=0.5):
    # FAISS embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # FAISS search
    _, faiss_top_indices = index.search(query_embedding, top_k)
    top_indices = faiss_top_indices[0]

    # BM25 search
    query_tokens = word_tokenize(query.lower())
    bm25_scores_all = bm25.get_scores(query_tokens)
    bm25_subset = np.array([bm25_scores_all[i] for i in top_indices])

    # Cosine similarity
    cosine_subset = cosine_similarity(query_embedding, df['embedding'].iloc[top_indices].tolist())[0]

    # Normalize scores
    bm25_subset /= (np.linalg.norm(bm25_subset) + 1e-10)
    cosine_subset /= (np.linalg.norm(cosine_subset) + 1e-10)

    # Combine using hybrid scoring
    hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset
    top_hybrid_indices = np.argsort(hybrid_scores)[::-1][:final_k]
    final_indices = [top_indices[i] for i in top_hybrid_indices]

    # Return final results as a single string
    return " ".join(df['clean_embedding_text'].iloc[final_indices].tolist())
