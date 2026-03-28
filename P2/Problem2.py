import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# === CONFIGURATION ===
TEXT_FILE = "../text_files/mcp_security.txt"  # Problem 2a: Your document
OUTPUT_EXCEL = "CS795_Retrieval_Results.xlsx"
OUTPUT_MARKDOWN = "Retrieval_Report.md"

CHUNK_SIZE = 100  # words per chunk
CHUNK_OVERLAP = 30  # words shared between chunks
TOP_K = 3  # Number of results to retrieve per query

# 1. INITIALIZE MODEL (Problem 2c)
print("Loading Transformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# 2. DEFINE CORE FUNCTIONS
def get_word_chunks(file_path, size, overlap):
    """Problem 2b: Load and split text into overlapping word-based chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    words = text.split()
    chunks = []

    # Iterate through words with overlap
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i: i + size])
        chunks.append(chunk)
        # Stop if we've reached the end
        if i + size >= len(words):
            break
    return chunks


def semantic_search(query, vector_store, k=3):
    """Problem 3e: Compute query embedding and retrieve top k using cosine similarity."""
    query_embedding = model.encode(query)

    # Extract all embeddings from the store into a single matrix
    all_embeddings = np.stack(vector_store['embedding'].values)

    # Calculate Cosine Similarity
    # util.cos_sim returns a matrix; [0] gives the similarities for our single query
    cos_scores = util.cos_sim(query_embedding, all_embeddings)[0]

    # Get top k indices
    top_results = cos_scores.argsort(descending=True)[:k]

    results = []
    for idx in top_results:
        i = idx.item()
        results.append({
            "Score": round(float(cos_scores[i]), 4),
            "Chunk_ID": vector_store.iloc[i]['chunk_id'],
            "Content": vector_store.iloc[i]['content']
        })
    return results


# === MAIN EXECUTION ===

# Step 2a & 2b: Load and Chunk
print(f"Chunking document: {TEXT_FILE}...")
chunks = get_word_chunks(TEXT_FILE, CHUNK_SIZE, CHUNK_OVERLAP)


# Step 2c & 2d: Generate Embeddings and Store
print(f"Generating embeddings for {len(chunks)} chunks...")
chunks_df = pd.DataFrame({'number':list(range(len(chunks))), 'chunk':chunks})
chunks_df.to_csv("chunks.csv")

embeddings = model.encode(chunks)

vector_store = pd.DataFrame({
    "chunk_id": range(1, len(chunks) + 1),
    "content": chunks,
    "embedding": list(embeddings)
})
vector_store.to_pickle("final_vector_store.pkl")
print("Vector store saved to final_vector_store.pkl")

# Step 3f: Run Example Queries and Collect Data for Report
example_queries = [
    "What is a Tool poisoning attack in MCP?",
    " What are the limitation of previous works on MCP servers",
    "What are the security best practices for MCP servers?"
]

report_list = []
print("\nRunning Retrieval Queries...")

for q in example_queries:
    matches = semantic_search(q, vector_store, k=TOP_K)
    for rank, match in enumerate(matches, 1):
        report_list.append({
            "Query": q,
            "Rank": rank,
            "Similarity Score": match["Score"],
            "Chunk ID": match["Chunk_ID"],
            "Retrieved Text": match["Content"]
        })

# === EXPORT RESULTS FOR PROFESSOR ===
report_df = pd.DataFrame(report_list)

# 1. Save to Excel (Professional Spreadsheet)
report_df.to_excel(OUTPUT_EXCEL, index=False)

# 2. Save to Markdown (Clean text report)
with open(OUTPUT_MARKDOWN, "w") as f:
    f.write("# CS 795/895: RAG Retrieval Report\n\n")
    for q in example_queries:
        f.write(f"### Query: {q}\n")
        subset = report_df[report_df['Query'] == q]
        f.write(subset[['Rank', 'Similarity Score', 'Retrieved Text']].to_markdown(index=False))
        f.write("\n\n---\n\n")

print(f"\nSUCCESS!")
print(f"- Processed {len(chunks)} chunks.")
print(f"- Results saved to {OUTPUT_EXCEL} and {OUTPUT_MARKDOWN}")
