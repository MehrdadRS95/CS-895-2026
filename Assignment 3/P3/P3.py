import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# --- 1. Setup & Load Data ---
load_dotenv()  # Loads your OPENAI_API_KEY from a .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading Vector Store and Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# Using your specific path
pkl_path = "/Assignment 3/P2/final_vector_store.pkl"
vector_store = pd.read_pickle(pkl_path)


# --- 2. Functions ---

def get_retrieved_context(query, k=3):
    """Retrieves the top k most relevant chunks."""
    query_vec = model.encode(query)
    all_embeddings = np.stack(vector_store['embedding'].values)
    similarities = util.cos_sim(query_vec, all_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:k]
    return vector_store.iloc[top_indices.tolist()]


def construct_problem3a_prompt(question, context_df):
    """Constructs the augmented prompt for the LLM."""
    context_text = ""
    for _, row in context_df.iterrows():
        context_text += f"--- START CHUNK (ID: {row['chunk_id']}) ---\n"
        context_text += f"{row['content']}\n"
        context_text += f"--- END CHUNK ---\n\n"

    return f"""SYSTEM INSTRUCTIONS:
You are a security research assistant. Use the provided context to answer the user question.
- Answer ONLY using the information in the context.
- If the answer isn't there, say you don't know.
- You MUST cite the Chunk ID for every fact you mention (e.g., "The model is vulnerable [ID: 2]").

RETRIEVED CONTEXT:
{context_text}

USER QUESTION:
{question}

FINAL RESPONSE:
"""


def generate_answer(prompt):
    """Sends the prompt to OpenAI and gets the cited response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-3.5-turbo or gpt-4o depending on preference
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Low temperature for factual accuracy (Problem 1d!)
    )
    return response.choices[0].message.content


# --- 3. Problem 3b: Execution for 5 Questions ---

queries = [
    "What is a tool poisoning attack in MCP?",
    "How does tool poisoning affect AI agents in the MCP ecosystem?",
    "What are the security issues in MCP servers?",
    "Explain MCP host.",
    "What are the primary and secondary defense layers in MCP eco-system"
]

final_results = []

print(f"\nProcessing {len(queries)} questions...")

q = "What is the primary defense layer in MCP security?"
context_chunks = get_retrieved_context(q)
first_prompt = construct_problem3a_prompt(question=q, context_df=context_chunks)

with open("README.md", "a") as f:
    f.write(f"\n## Experiment Result\n{first_prompt}\n")

print("Result successfully written to README.md")

for i, q in enumerate(queries, 1):
    print(f"Answering Question {i}...")

    # Step A: Retrieve
    context_chunks = get_retrieved_context(q)

    # Step B: Construct Prompt
    augmented_prompt = construct_problem3a_prompt(q, context_chunks)

    # Step C: Generate Answer
    answer = generate_answer(augmented_prompt)

    final_results.append({
        "Question": q,
        "Answer": answer
    })


def generate_answer_no_retrieval(question):
    """
    Problem 3c: Generates an answer using only the LLM's internal knowledge.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful security assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.1  # Keep temperature consistent for fair comparison
    )
    return response.choices[0].message.content


# --- 2. Execution ---
no_retrieval_results = []

print("Generating answers WITHOUT retrieval context...")

for i, q in enumerate(queries, 1):
    print(f"Processing Question {i}...")
    answer = generate_answer_no_retrieval(q)
    no_retrieval_results.append({
        "Question": q,
        "Answer_No_Retrieval": answer
    })

# --- 3. Output & Saving ---
print("\n--- NON-RETRIEVAL ANSWERS ---")
for res in no_retrieval_results:
    print(f"\nQ: {res['Question']}")
    print(f"A: {res['Answer_No_Retrieval'][:200]}...")  # Printing just the start for brevity

# Save to CSV for easy comparison in Excel
df_rag_based = pd.DataFrame(final_results)
df_rag_based.to_csv("P3_b_rag_based_results.csv")

df_no_retrieval = pd.DataFrame(no_retrieval_results)
df_comparison = pd.merge(df_no_retrieval, df_rag_based, on='Question', how='inner')
df_comparison.to_csv("P3_c_d_result_comparison.csv", index=False)

# --- 4. Final Output ---

print("\n" + "=" * 50)
print("FINAL RAG ANSWERS")
print("=" * 50)

for res in final_results:
    print(f"\nQ: {res['Question']}")
    print(f"A: {res['Answer']}")
    print("-" * 30)

# Save to a text file for your report
with open("Problem3b_Final_Answers.txt", "w") as f:
    for res in final_results:
        f.write(f"Question: {res['Question']}\n")
        f.write(f"Answer: {res['Answer']}\n")
        f.write("-" * 50 + "\n")
