# retriever.py

import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os



from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np

def validate_resources(index, combined_embeddings, combined_metadata, combined_corpus):
    # Basic sanity checks
    assert combined_embeddings.shape[0] == len(combined_metadata), \
        f"Embeddings count ({combined_embeddings.shape[0]}) and metadata count ({len(combined_metadata)}) mismatch"
    assert combined_embeddings.shape[0] == index.ntotal, \
        f"FAISS index size ({index.ntotal}) and embeddings count ({combined_embeddings.shape[0]}) mismatch"

    # Optional: check metadata type and BM25 corpus type
    assert isinstance(combined_metadata, list), "Metadata should be a list"
    assert all(isinstance(m, dict) for m in combined_metadata), "Each metadata item should be a dict"
    assert isinstance(combined_corpus, list), "BM25 corpus should be a list"
    assert all(isinstance(doc, list) for doc in combined_corpus), "Each BM25 corpus entry should be a list of tokens"







# Load everything once


# def load_resources():
#     base_path = os.path.join(os.path.dirname(__file__), "files")

#     index_path = os.path.join(base_path, "faiss_index.bin")
#     embeddings_path = os.path.join(base_path, "combined_embeddings.npy")
#     metadata_path = os.path.join(base_path, "metadata.pkl")
#     bm25_corpus_path = os.path.join(base_path, "bm25_corpus.pkl")

#     # Load FAISS index
#     index = faiss.read_index(index_path)

#     # Load combined embeddings
#     combined_embeddings = np.load(embeddings_path)

#     # Load metadata
#     with open(metadata_path, "rb") as f:
#         combined_metadata = pickle.load(f)

#     # Load BM25 corpus
#     with open(bm25_corpus_path, "rb") as f:
#         combined_corpus = pickle.load(f)
    
#     # Validate loaded resources
#     validate_resources(index, combined_embeddings, combined_metadata, combined_corpus)

#     bm25 = BM25Okapi(combined_corpus)
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     return model, index, combined_embeddings, combined_metadata, bm25


# Run hybrid retrieval
# def retrieve_relevant_text(query, model, index, combined_embeddings, combined_metadata, bm25, top_k=100, final_k=5, alpha=0.5):
#     query_embedding = model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(query_embedding)

#     _, faiss_top_indices = index.search(query_embedding, top_k)

#     query_tokens = word_tokenize(query.lower())
#     bm25_scores_all = bm25.get_scores(query_tokens)

#     bm25_subset = np.array([bm25_scores_all[i] for i in faiss_top_indices[0]])
#     cosine_subset = cosine_similarity(query_embedding, combined_embeddings[faiss_top_indices[0]])[0]

#     bm25_subset /= np.linalg.norm(bm25_subset) + 1e-10
#     cosine_subset /= np.linalg.norm(cosine_subset) + 1e-10

#     hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset

#     top_hybrid_indices = np.argsort(hybrid_scores)[::-1][:final_k]
#     final_indices = [faiss_top_indices[0][i] for i in top_hybrid_indices]

#     results = [combined_metadata[i] for i in final_indices]
#     top_texts = [r["text"] for r in results]

#     return top_texts if top_texts else ""
############################################################################################

# def retrieve_relevant_text(query, model, client, collection_name, combined_embeddings, combined_metadata, bm25, top_k=50, final_k=5, alpha=0.5):
#     # Step 1: Encode the query
#     query_vector = model.encode(query).tolist()
    
#     # Step 2: Search Qdrant
#     results = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     # Step 3: Extract indices of retrieved vectors
#     qdrant_indices = [int(hit.id) for hit in results]
    
#     if not qdrant_indices:
#         return ""

#     # Step 4: Compute BM25 and cosine similarity for hybrid scoring
#     query_tokens = word_tokenize(query.lower())
#     bm25_scores_all = bm25.get_scores(query_tokens)

#     bm25_subset = np.array([bm25_scores_all[i] for i in qdrant_indices])
#     cosine_subset = cosine_similarity(
#         [query_vector],
#         combined_embeddings[qdrant_indices]
#     )[0]

#     # Normalize
#     bm25_subset /= np.linalg.norm(bm25_subset) + 1e-10
#     cosine_subset /= np.linalg.norm(cosine_subset) + 1e-10

#     # Step 5: Combine scores
#     hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset
#     top_hybrid_indices = np.argsort(hybrid_scores)[::-1][:final_k]
#     final_indices = [qdrant_indices[i] for i in top_hybrid_indices]

#     # Step 6: Return top metadata/texts
#     results = [combined_metadata[i] for i in final_indices]
#     top_texts = [r["text"] for r in results]

#     return top_texts if top_texts else ""





###############################################

# def load_resources():
#     import os
#     import pickle
#     from sentence_transformers import SentenceTransformer
#     from rank_bm25 import BM25Okapi

#     base_path = os.path.join(os.path.dirname(__file__), "files")
#     bm25_corpus_path = os.path.join(base_path, "bm25_corpus.pkl")

#     # Load BM25 corpus
#     with open(bm25_corpus_path, "rb") as f:
#         combined_corpus = pickle.load(f)

#     # Initialize BM25
#     bm25 = BM25Okapi(combined_corpus)

#     # Load SentenceTransformer model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     return model, bm25




# def retrieve_relevant_text(query, model, client, collection_name, bm25, top_k=50, final_k=5, alpha=0.5):
#     query_vector = model.encode(query).tolist()
    
#     results = client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     if not results:
#         return ""

#     payloads = [hit.payload for hit in results]
#     qdrant_ids = [int(hit.id) for hit in results]
#     qdrant_texts = [p["text"] for p in payloads]

#     query_tokens = word_tokenize(query.lower())
#     bm25_scores_all = bm25.get_scores(query_tokens)
#     bm25_subset = np.array([bm25_scores_all[i] for i in qdrant_ids])  # ✅ safer fallback

#     #doc_vectors = np.array([p["embedding"] for p in payloads])
#     cosine_subset = cosine_similarity([query_vector], doc_vectors)[0]

#     # Normalize
#     bm25_subset /= np.linalg.norm(bm25_subset) + 1e-10
#     cosine_subset /= np.linalg.norm(cosine_subset) + 1e-10

#     hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset
#     top_indices = np.argsort(hybrid_scores)[::-1][:final_k]
#     top_texts = [payloads[i]["text"] for i in top_indices]

#     return top_texts if top_texts else ""



##########################################################################################

##best
# def load_resources():
#     import os
#     import pickle
#     import numpy as np
#     from rank_bm25 import BM25Okapi
#     from sentence_transformers import SentenceTransformer

#     base_path = os.path.join(os.path.dirname(__file__), "files")

#     # File paths
#     embeddings_path = os.path.join(base_path, "combined_embeddings.npy")
#     metadata_path = os.path.join(base_path, "metadata.pkl")
#     bm25_corpus_path = os.path.join(base_path, "bm25_corpus.pkl")

#     # Load combined embeddings
#     combined_embeddings = np.load(embeddings_path)

#     # Load metadata
#     with open(metadata_path, "rb") as f:
#         combined_metadata = pickle.load(f)

#     # Load BM25 tokenized corpus
#     with open(bm25_corpus_path, "rb") as f:
#         combined_corpus = pickle.load(f)

#     # Sanity check
#     assert len(combined_metadata) == combined_embeddings.shape[0] == len(combined_corpus), "Mismatch in data sizes"

#     # Create BM25 model
#     bm25 = BM25Okapi(combined_corpus)

#     # Load embedding model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     return model, combined_embeddings, combined_metadata, bm25


#########################################################################








#########################
#work succesfully
    # def retrieve_relevant_text(query, model, client, collection_name, combined_embeddings, combined_metadata, bm25, top_k=50, final_k=5, alpha=0.5):

    #     # Step 1: Encode the query
    #     query_vector = model.encode(query)

    #     # Step 2: Search Qdrant
    #     results = client.search(
    #         collection_name=collection_name,
    #         query_vector=query_vector.tolist(),
    #         limit=top_k
    #     )

    #     if not results:
    #         return ""

    #     # Step 3: Extract IDs and metadata
    #     qdrant_ids = [int(hit.id) for hit in results]
    #     payloads = [hit.payload for hit in results]

    #     # Step 4: BM25 scoring
    #     query_tokens = word_tokenize(query.lower())
    #     bm25_scores_all = bm25.get_scores(query_tokens)
    #     bm25_subset = np.array([bm25_scores_all[i] for i in qdrant_ids])

    #     # Step 5: Cosine similarity with local embeddings
    #     doc_vectors = combined_embeddings[qdrant_ids]
    #     cosine_subset = cosine_similarity([query_vector], doc_vectors)[0]

    #     # Step 6: Normalize and combine
    #     bm25_subset /= np.linalg.norm(bm25_subset) + 1e-10
    #     cosine_subset /= np.linalg.norm(cosine_subset) + 1e-10

    #     hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset
    #     top_indices = np.argsort(hybrid_scores)[::-1][:final_k]

    #     # Step 7: Return top texts
    #     top_texts = [combined_metadata[qdrant_ids[i]]["text"] for i in top_indices]
    #     return top_texts if top_texts else ""
#########################################################
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize

def retrieve_relevant_text(query, model, client, collection_name, bm25, top_k=50, final_k=5, alpha=0.5):
    # Step 1: Encode the query
    query_vector = model.encode(query)

    # Step 2: Search Qdrant (retrieve both payload and vectors)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_vectors=True  # ✅ Ensures vector is returned
    )

    if not results:
        return ""

    # Step 3: Extract vectors and payloads from Qdrant
    doc_vectors = np.array([hit.vector for hit in results])
    payloads = [hit.payload for hit in results]

    # Step 4: BM25 scoring
    query_tokens = word_tokenize(query.lower())
    bm25_scores_all = bm25.get_scores(query_tokens)

    # Use Qdrant hit.id to index BM25 scores
    qdrant_ids = [int(hit.id) for hit in results]
    bm25_subset = np.array([bm25_scores_all[i] for i in qdrant_ids])

    # Step 5: Cosine similarity
    cosine_subset = cosine_similarity([query_vector], doc_vectors)[0]

    # Step 6: Normalize and combine scores
    bm25_subset /= np.linalg.norm(bm25_subset) + 1e-10
    cosine_subset /= np.linalg.norm(cosine_subset) + 1e-10

    hybrid_scores = alpha * cosine_subset + (1 - alpha) * bm25_subset
    top_indices = np.argsort(hybrid_scores)[::-1][:final_k]

    # Step 7: Return top texts from payloads
    top_texts = [payloads[i]["text"] for i in top_indices]

    return top_texts if top_texts else ""



def load_resources():
    import os
    import pickle
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    base_path = os.path.join(os.path.dirname(__file__), "files")

    # Load BM25 tokenized corpus
    bm25_corpus_path = os.path.join(base_path, "bm25_corpus.pkl")
    with open(bm25_corpus_path, "rb") as f:
        combined_corpus = pickle.load(f)

    # Create BM25 model
    bm25 = BM25Okapi(combined_corpus)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return model, bm25
