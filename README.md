
# CS795/895 — Large Language Model Architectures and Applications

## Assignment 2
### Mehrdad Rostamzadeh

## Problem 1
This assignment studies LLM inference and retrieval-augmented generation (RAG). You
may use any accessible models or libraries (e.g., GPT-4o Mini, Claude Haiku, Gemini Flash,
LLaMA-3-8B-Instruct, Mistral-7B-Instruct, Qwen-2.5-7B-Instruct, Hugging Face tools).

**Problem 1**

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
