<h1 align="center"> CS795/895 — Large Language Model Architectures and Applications </h1>


<h2 align="center"> Homework Assignment 5 </h2>

**This assignment studies the Transformer architecture, including its theoretical founda-
tions, implementation, and analysis of attention mechanisms.
You may use any accessible tools or libraries (e.g., Hugging Face Transformers, PyTorch,
TensorFlow, Google Colab).**

## Table of Contents

- [Problem 1](#Problem-1)
- [Problem 2](#installation)
- [Problem 3](#usage)
- [Problem 4](#features)
- [Problem 5](#contributing)

## Problem 1

#### a. Explain the main differences between recurrent neural networks (RNNs), long short-term memory networks (LSTMs), and Transformers.

1. **RNN**

- **How they work:** RNNs process data sequentially, one step (or word) at a time. As they read each word, they update
  a "hidden state" which acts as a running memory of the sentence so far.
- **The Advantage:** They naturally understand the order of sequences because of their step-by-step design.
- **The flaw:** Standard RNNs have a very "short memory." When processing long sentences, the information from the
  beginning of the sentence gradually fades away or gets diluted by the time the model reaches the end.

2. **Long Short-Term Memory Networks (LSTMs)**

- **How they work:** LSTMs are a specialized, advanced version of RNNs. Instead of a simple hidden state, they use a
  complex "cell state" accompanied by three gates: an Input Gate, a Forget Gate, and Output Gate.

- **The Advantage:** These gates act like a smart filter, learning exactly what information to keep in long-term memory
  and what irrelevant information to forget. This effectively solves the vanishing gradient problem of vanilla RNNs,
  allowing them to handle much longer sequences.

- **The Flaw:** They are computationally heavy and complex. Furthermore, because they are still recurrent (processing
  word-by-word), they still suffer from the same severe bottleneck: they cannot be parallelized efficiently during
  training.

3. **Transformers**

- **How they work:** Transformers completely abandon the recurrent, step-by-step processing. Instead, they process the
  entire sequence simultaneously using a mechanism called Self-Attention. This allows the model to look at every single
  word in a sentence at the exact same time and mathematically score how each word relates to every other word,
  regardless of how far apart they are.

- **The Advantage:** 1. Infinite "Memory": They capture long-range dependencies perfectly because the distance between
  the first word and the last word is always just one step via the attention mechanism.


- **The Flaw:** The self-attention mechanism scales quadratically (O(N^2)). If you double the length of the input text,
  the computational memory required quadruples, making it very expensive to process extremely long documents.
  Additionally, because they process everything at once, they require artificial "positional encodings" to understand
  the original order of the words.

#### b. Discuss why Transformers have become the dominant architecture for many natural language processing tasks.

1. **The Rise of Transfer Learning and Foundation Models**
   Transformers proved to be exceptionally good at "unsupervised pre-training." A single, massive Transformer can read
   billions of web pages to learn the fundamental statistical structure of human language. Once this massive "foundation
   model" is trained, it can be easily and cheaply fine-tuned for hundreds of specific downstream tasks—such as
   translation, summarization, coding, or sentiment analysis—requiring very little task-specific data.

3. **Superior Long-Range Context**
   Human language heavily relies on context. A pronoun at the end of a long paragraph might refer to a subject
   introduced in the very first sentence. While LSTMs attempted to solve this, they still struggled with very long gaps.
   The Transformer's self-attention mechanism connects every word directly to every other word in the sequence. This
   creates a near-perfect understanding of context, regardless of how far apart the related words are.

#### c. Define self-attention and explain how it differs from traditional attention mechanisms used in sequence models.

**Self Attention:**
Self-attention is a mechanism in neural networks where a sequence of elements attends to itself to compute a
representation for each element based on the relationships between all elements in the sequence.

- Every token in a sequence looks at all other tokens in that same sequence.
  It assigns weights (attention scores) to other tokens to decide how much each contributes to its new representation

Traditional Attention (Encoder-Decoder Attention)
Traditional attention mechanisms (such as Bahdanau or Luong attention) were originally designed for recurrent
sequence-to-sequence (Seq2Seq) models, primarily to improve machine translation. In these models, attention acts as a
bridge between two distinct sequences. As the decoder generates a translated output word-by-word (e.g., in French), it
uses the traditional attention mechanism to "look back" at the encoder's hidden states (the English source sequence) to
figure out which foreign word it should focus on at that exact moment.

Key Differences

**Traditional Attention:** Operates between two different sequences. It connects the input sequence to the output
sequence (
Encoder to Decoder).

**Self-Attention:** Operates within a single sequence. It connects elements of the input sequence to other elements of
the
same input sequence (or output to output) to build a richer contextual understanding of the text itself.

Architectural Dependency:

**Traditional Attention:** Was built as an add-on for RNNs and LSTMs. It was created to solve the bottleneck of trying
to
compress a whole sentence into one fixed-size vector, but it still fundamentally relied on the step-by-step processing
of the RNN.

**Self-Attention:** In Transformers, self-attention completely replaces the RNN structure. It does not process text
step-by-step; instead, it evaluates the relationships between all words simultaneously, which is the core driver of
modern AI's massive parallelization and speed.

##### d. Describe the purpose of positional encodings in the Transformer architecture.

Because Transformers process all words simultaneously rather than step-by-step, they naturally lose track of word order.
Positional encodings solve this by attaching a mathematical "tag" (using sine and cosine functions) to each word's data
before processing. This tag tells the model exactly where the word is located in the sentence, allowing the Transformer
to understand both the meaning and the correct sequence of the words without sacrificing its fast, parallel processing
speed.

#### e. Explain the concept of multi-head attention and why it improves model performance.

**The Concept of Multi-Head Attention**
Instead of performing a single self-attention calculation, Transformers use a mechanism called "multi-head attention."
This means the model runs the self-attention process multiple times in parallel (these parallel processes are called "
heads"). Each head has its own independently learned set of weights (queries, keys, and values). After all the heads
have processed the sequence simultaneously, their individual outputs are concatenated (stitched back together) and
linearly transformed to produce the final output.

**Why it Improves Model Performance**

***Multiple Perspectives (Representation Subspaces):*** With a single attention head, the model might average out or lose
specific contextual details because it is forced to focus on only one dominant relationship at a time. Multi-head
attention allows different heads to specialize in learning different types of relationships. For example, when analyzing
a sentence, one attention head might focus heavily on tracking pronouns (who "it" refers to), another might focus on the
grammatical structure (verb-subject relationships), and a third might track emotional sentiment.

***Richer Context:*** By combining these various independent "perspectives" back together, the model gains a much
richer, multidimensional understanding of the text. It allows the model to jointly attend to information from different
representation subspaces at different positions, significantly boosting its reasoning capabilities, accuracy, and
ability to handle complex language structures.

#### f. Identify and briefly describe the key components of the Transformer architecture introduced in the “Attention Is All You Need” paper.




