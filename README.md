# Kaggle & ML Projects

A collection of personal machine learning and data science projects, including credit risk modelling, fraud detection, NLP, and a **fully local Retrieval‑Augmented Generation (RAG) pipeline** with automated evaluation.

## 📁 Repository Structure

| Folder / File | Description |
|---------------|-------------|
| `credit_card_application` | Credit card approval prediction using classical ML. |
| `credit_card_fraud_detection` | Fraud detection with imbalanced dataset handling. |
| `fake_and_real_news` | Text classification for fake news detection. |
| `twitter_sentiment_analysis` | Entity‑level sentiment analysis on tweets using a fine‑tuned BERT. |
| `single_topic_RAG_evaluation` | **End‑to‑end RAG pipeline with Ollama & RAGAS** (see below). |
| `.gitignore` | Common Python ignores. |
| `requirements.txt` | Python dependencies for the repository. |

## 🧪 Highlight: Single‑Topic RAG Evaluation

This project builds a **completely local RAG system** using:

- [Ollama](https://ollama.com) + `llama3.1:8b` as the generator
- `all-MiniLM-L6-v2` embeddings stored in Chroma
- LangChain for orchestration
- [RAGAS](https://docs.ragas.io) for quantitative evaluation

It uses the [Single‑Topic RAG Evaluation Dataset](https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset/data) and systematically experiments with chunking strategies, retrieval parameters, re‑ranking, and hybrid search. The notebook includes a full experiment log and final results table, making it easy to understand trade‑offs and performance bottlenecks.

**Key findings from the experiments:**

- Cross‑encoder re‑ranking alone lifted context precision from **0.28 → 0.49** and recall from **0.32 → 0.54**.
- Larger chunk sizes improved faithfulness but hurt retrieval granularity.
- Hybrid search (BM25 + vector) degraded performance due to naïve merging; reciprocal rank fusion is recommended for future work.
