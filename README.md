# CS795/895 — Large Language Model Architectures and Applications

## Assignment 2

### Mehrdad Rostamzadeh

This assignment studies LLM inference and retrieval-augmented generation (RAG). You
may use any accessible models or libraries (e.g., GPT-4o Mini, Claude Haiku, Gemini Flash,
LLaMA-3-8B-Instruct, Mistral-7B-Instruct, Qwen-2.5-7B-Instruct, Hugging Face tools).

## Problem 1

**a. Select 5 prompts requiring short but nontrivial responses.**

```
prompts = [
    "Define a 'Large Language Model' (LLM) in exactly one sentence.",
    "What is a 'prompt' in the context of AI?",
    "Write a Python function that returns the sum of two numbers.",
    "What does it mean when an AI 'hallucinates'?",
    "What is a 'token' in natural language processing?. Briefly explain it"
]
```

**b. Run them with at least two inference settings (e.g., different temperature values).**

The table below shows selected prompts with two different responses, the response 1 is generated with temperature 0.1,
and the response 2 is generated with temperature 0.9.


  <table>
  <thead>
    <tr>
      <th>Number</th>
      <th>Prompt</th>
      <th>Response 1</th>
      <th>Response 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Define a 'Large Language Model' (LLM) in exactly one sentence.</td>
      <td>A Large Language Model (LLM) is an advanced artificial intelligence system designed to understand, generate, and manipulate human language by leveraging vast amounts of text data and complex neural network architectures.</td>
      <td>A Large Language Model (LLM) is a type of artificial intelligence that uses deep learning techniques to process and generate human-like text by predicting the next word in a sequence based on vast amounts of training data.</td>
    </tr>
    <tr>
      <td>2</td>
      <td>What is a 'prompt' in the context of AI?</td>
      <td>
        <p>In the context of AI, particularly in natural language processing and machine learning, a "prompt" refers to the input or instruction given to an AI model to elicit a response or generate content. Prompts can take various forms, such as questions, statements, or specific tasks, and they guide the AI in producing relevant and coherent outputs.</p>
        <p>For example, if you provide a prompt like "Write a short story about a dragon," the AI will generate a narrative based on that instruction.</p>
      </td>
      <td>
        <p>In the context of AI, particularly in natural language processing and machine learning, a "prompt" refers to the input provided to a model to elicit a specific response or behavior. Prompts can take various forms, such as text, questions, or instructions.</p>
        <p>For example, in a conversation with a language model, a user might provide a prompt like "Tell me a story about a dragon." The model uses this prompt to generate a narrative that aligns with the theme of a dragon.</p>
      </td>
    </tr>
    <tr>
      <td>3</td>
      <td>Write a Python function that returns the sum of two numbers.</td>
      <td>
        <pre><code>def sum_two_numbers(a, b):
    return a + b</code></pre>
      </td>
      <td>
        <pre><code>def sum_two_numbers(num1, num2):
    return num1 + num2</code></pre>
      </td>
    </tr>
    <tr>
      <td>4</td>
      <td>What does it mean when an AI 'hallucinates'?</td>
      <td>When an AI "hallucinates," it refers to the phenomenon where the AI generates information that is false, misleading, or nonsensical, despite sounding plausible.</td>
      <td>
        <i>Note: The data provided for this cell appears to describe "Tokens" rather than "Hallucinations." You may want to verify this content.</i>
      </td>
    </tr>
    <tr>
      <td>5</td>
      <td>What is a 'token' in natural language processing?</td>
      <td>
        <p>In natural language processing (NLP), a "token" refers to a single unit of text that has been extracted from a larger body of text during the process of tokenization.</p>
        <p>For example, in the sentence "The cat sat on the mat," the tokens could be: ["The", "cat", "sat", "on", "the", "mat"].</p>
      </td>
      <td>
        <p>In natural language processing (NLP), a "token" is a unit of text that is processed as a single entity. Tokens can be words, characters, phrases, or subwords.</p>
        <p>Tokenization is the initial step in NLP tasks, as it breaks down text into manageable pieces for further analysis.</p>
      </td>
    </tr>
  </tbody>
</table>      


**c. Record outputs and compare determinism, diversity, and factual consistency.**

***Inference Comparison Analysis***

***Determinism (High):*** Both responses were highly predictable, using the exact same examples, "dragon story" for
prompts and "The cat sat on the mat" for tokens. The Python functions also used identical logic and naming conventions.
***Diversity (Low):*** There was minimal variation between the two settings. Changes were limited to minor word swaps (
e.g., "type of AI" vs. "advanced AI system") or simple variable renaming in the code from a, b to num1, num2.
***Factual Consistency (Mixed):*** Response 1 was fully accurate. However, Response 2 failed Prompt 4 entirely by
providing the definition of a "token" instead of explaining "hallucinations," representing a significant hallucination
or instruction-following error.

**d. Explain how decoding choices influence generation.**

Decoding choices like Temperature acts like a creativity option that tells the AI how much risk to take when AI wants to
select the next word. When the temperature is low (in this problem 0.1), the AI output becomes almost deterministic,
meaning it strictly picks the most mathematically likely word every time to ensure the answer is consistent and factual.
On the other hand, when the temperature is high (in this problem 0.9), it increases the diversity of the output by
allowing the AI to select less common words, which makes the responses feel more unique and creative, but also less
predictable.

**e. Give one application where deterministic inference is preferable and one where diversity
is useful.**

**Code generation** is an application that requires low diversity and therefore,a low temperature to ensure the
syntactic correctness and reliable execution. Increasing the temperature may increase creativity but often results in
broken code.
**Creative writing and brainstorming** are two applications that benefit from a high temperature. A higher temperature
increases creativity and enables us to explore more unique ideas, metaphors, and slogans beyond the most statistically
likely outputs.

## Problem 2

**a. Select a text document or small document collection.**

You can find the security documentation here:

[MCP Security Document](text_files/mcp_security.txt)

**b. Load the document(s) and split the text into overlapping chunks.**

You can see the chunks here:

[Document split chunks](P2/chunks.csv)

**c. Generate embeddings for each chunk using a transformer or sentence-embedding model.**

Embedding for each chunk is generated using a transformer model named "all-MiniLM-L6-v2"
Ant it is stored in "P2/final_vector_store.pkl"

**d. Store each chunk and its embedding in a simple vector store.**

The embedding file is stored in a .pkl file in the P2 folder. This will used for the next problems.

**e. For a user query, compute its embedding and retrieve the top k chunks using cosine similarity.**

**f. Report retrieved chunks for at least three example queries.**

Example queries used for this part:

```
example_queries = [
    "What is a Tool poisoning attack in MCP?",
    " What are the limitation of previous works on MCP servers",
    "What are the security best practices for MCP servers?"
]

```

You can see the result of embedding and top k chunks here:

[Retrieval Report](P2/Retrieval_Report.md)

## Problem 3

**Problem 3**

**a. Construct a prompt that includes:**

- The question
- Retrieved context
- An instruction to answer using the context
- An instruction to cite the retrieved passages

## Experiment Result

SYSTEM INSTRUCTIONS:
You are a security research assistant. Use the provided context to answer the user question.

- Answer ONLY using the information in the context.
- If the answer isn't there, say you don't know.
- You MUST cite the Chunk ID for every fact you mention (e.g., "The model is vulnerable [ID: 2]").

RETRIEVED CONTEXT:
--- START CHUNK (ID: 1) ---
Limitations of Existing MCP Security Research Despite recent progress in analyzing the security of the Model Context
Protocol (MCP), existing work exhibits several important limitations: - Attack-centric focus. Most studies prioritize
identifying attack vectors or measuring vulnerability and attack success rates, but do not specify where defenses should
be deployed within the MCP architecture or which components are responsible for enforcement. - Lack of defense placement
guidance. Existing taxonomies and mitigation efforts rarely distinguish between primary (earliest feasible) and
secondary (fallback or compensatory) defense layers, making it difficult to reason about defense ordering and
composition. - Partial defense coverage. Proposed
--- END CHUNK ---

--- START CHUNK (ID: 18) ---
layer operates closest to the attack source, aiming to stop malicious inputs, behaviors, or configurations before they
can influence downstream components. --- ### Secondary Defense Layer The secondary defense layer provides
defense-in-depth by limiting the impact and propagation of an attack when the primary defense layer fails, is
misconfigured, or is deliberately bypassed. Rather than preventing initial compromise, this layer focuses on
containment, mitigation, and recovery. To clarify how primary and secondary defense layers differ in practice, a rug
pull attack provides a concrete example. A malicious MCP tool can successfully pass pre-execution trust decisions at the
registry and approval
--- END CHUNK ---

--- START CHUNK (ID: 17) ---
correctly yet still produce unsafe outcomes. As a result, this layer is fundamental for enforcing trust boundaries,
provenance guarantees, and long-term ecosystem safety rather than runtime correctness alone. --- ### Primary Defense
Layer The primary defense layer refers to the first line of protection designed to prevent an attack from occurring in
the first place, or to block it at the earliest possible point of interaction with the system. This layer operates
closest to the attack source, aiming to stop malicious inputs, behaviors, or configurations before they can influence
downstream components. --- ### Secondary Defense Layer The secondary defense layer
--- END CHUNK ---

USER QUESTION:
What is the primary defense layer in MCP security?

FINAL RESPONSE:

**b. Answer at least 5 questions using retrieved context.**

The selected questions are:

```
queries = [
    "What is a tool poisoning attack in MCP?",
    "How does tool poisoning affect AI agents in the MCP ecosystem?",
    "What are the security issues in MCP servers?",
    "Explain MCP host.",
    "What are the primary and secondary defense layers in MCP eco-system"
]
```

You can find the link to the responses here:

[Problem3_a](P3/P3_b_rag_based_results.csv)

**c. Also obtain answers without retrieval.**

You can find the link to the responses with and without retrival here:

[Problem3_c_d](P3/P3_c_d_result_comparison.csv)

**d. Compare grounded and ungrounded responses (correctness, specificity, hallucinations).**

Based on the result in
[Problem3_c_d](P3/P3_c_d_result_comparison.csv)
we can claim that **Grounded responses** prioritize Faithfulness to the source material, ensuring that in high-stakes
fields like medical research or MCP security, the model doesn't drift into generalities or invent facts. Ungrounded
responses may be more 'fluent' or 'conversational,' but they lack the traceability and specific technical accuracy
required for professional RAG systems

***Grounded Approach***: A perfectly grounded system should look at the token context, see it doesn't mention
hallucinations, and say: "I do not have enough information in the provided context to answer."

***Ungrounded Approach***: The model sees the word "hallucinate" and answers from its own memory. While the answer is "
correct" about AI, it is a system failure because it ignored the provided data.Grounded Approach**: A perfectly grounded
system should look at the token context, see it doesn't mention hallucinations, and say: "I do not have enough
information in the provided context to answer."

***Ungrounded Approach***: The model sees the word "hallucinate" and answers from its own memory. While the answer is "
correct" about AI, it is a system failure because it ignored the provided data.

**e. Briefly discuss when retrieval improves answers.**

***Freshness***: Overcomes the knowledge cutoff by providing real-time data or news.

***Private Data***: Grants the model access to internal documents or project files (like your MCP security docs) it
wasn't trained on.

***Factuality***: Significantly reduces hallucinations by grounding responses in provided text rather than "guessing."

***Verifiability***: Enables citations (e.g., "[Chunk 1]"), allowing users to audit and trust the information.

***Domain Specificity***: Ensures accuracy for niche topics—like specific medical interactions or rare code
libraries—that a general model might forget.

## Problem 4

a. Identify at least two incorrect or weak answers from your system.

**First weakness (Query 1 and 2)**

- Because the RAG-provided context is limited and based on a small set of documentation, the responses may lack
  accuracy, and similar queries are likely to yield repetitive or identical answers.
- As the context provided by RAG is not comprehensive and it is a small documentation, the response is not accurate and
  similar queries may receive same responses.

**Second weakness (Based on query 4)**
The system responds with "I dont't know", which is not desirable. Although RAG is expected to provide more accurate
results, no relevant information is received in for this query.

**b. For each, determine whether the issue arises from:**

- chunking
- embeddings
- retrieval
- prompting
- generation

First weakness is mainly because of "Retrieval". My retriever is probably pulling the exact same "Top 3" chunks for every query that mentions "security." If it isn't finding diverse information, the AI has nothing new to say.
Moreover, It could be a problem arising from embeddings: If the embedding model isn't sensitive enough, it sees two different questions (like "How do I secure MCP?" and "What are MCP threats?") as the exact same "math vector." Because the math looks the same, it grabs the same files.

**Second weakness**
The issue: The system gives up because it couldn't find any relevant data for Query 4.

***Primary reasons***:

***Retrieval:*** This is a direct "search failure." The system went into the library to find information on your query and came back empty-handed. Either the "keywords" didn't match or the search algorithm isn't strong enough.

***Prompting:*** Ironically, the "I don't know" part is actually a Prompting success. You told the AI to say that if it was confused (to prevent lying). The "weakness" is that the prompt is so strict it doesn't allow the AI to even try to help if the retrieval wasn't perfect.

***Chunking:*** If the answer is actually in your document but was "cut in half" during the chunking process, the retriever might not recognize it as a match, causing the system to miss it entirely.

c. Explain how chunk size and overlap affect retrieval.

d. Suggest two improvements to the system.

e. If relevant, briefly relate RAG to your course project.