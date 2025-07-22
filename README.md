# Retrieval-Augmented Generation QA for Pittsburgh & CMU

This repository contains a Retrieval-Augmented Generation (RAG) system designed to answer factual questions about Pittsburgh and Carnegie Mellon University.

## Overview

- **RAG System**: Combines document retrieval (using FAISS and dense embeddings) with answer generation from the Gemma 3 4B LLM (Ollama).
- **Knowledge Base**: Compiled and cleaned from public Pittsburgh/CMU-related web sources including Wikipedia, city government, and cultural/sports sites. Text and key PDFs are chunked for retrieval.
- **Prompting**: Strict prompt format—answers use only retrieved context; model says "I don't know" if answer is not found.
- **Evaluation**: Tested on a 100-question self-annotated set; metrics are Exact Match (EM) and token-level F1.

## Main Features

- **Embedding Models**: Supports both MiniLM and `thenlper/gte-small` for improved retrieval quality.
- **3 Pipeline Variations**:
  - V1: MiniLM, standard prompt
  - V2: MiniLM, strict concise prompt
  - V3: GTE-small, strict concise prompt (best)
- **Closed-Book Baseline**: LLM answers without retrieval for comparison.

## Results

| System Variant   | EM (%) | F1 (%) | Time (s) |
|------------------|--------|--------|----------|
| V1: MiniLM + Standard   |   46   |  69.9   |  ~3.0   |
| V2: MiniLM + Strict     |   58   |  75.5   |  ~2.8   |
| V3: GTE-small + Strict  |   58   |  78.1   |  ~2.8   |
| Closed-Book LLM         |   6    |  20.2   |  ~1.7   |

- **Best Variant (V3)**: EM 58%, F1 78%.
- **Closed-book model** underperforms RAG (EM 6%, F1 20%).

## Usage

1. **Prepare Knowledge Base**: Place cleaned `.txt` and extracted content in `knowledge_base_raw/`.
2. **Build Index & Run Pipeline**:
    - Run `rag_system_gemma.py` to build the retrieval and QA chain.
    - Run `evaluate_gemma_rag.py` for automated testing and metrics.
    - Adjust `evaluate_closed_book.py` for LLM-only baseline.
3. **Inspect Results**: Evaluation scores are output in `.json` for summary and analysis.

## Notes

- "I don't know" is returned for out-of-knowledge or unsupported questions—resulting in honest but incomplete coverage.
- Further improvements may include smarter retrieval, better chunking, and larger/recent LLMs.

## Requirements

- Python 3.8+
- Ollama (with Gemma 3 4B model)
- LangChain, FAISS, HuggingFace Transformers
- BeautifulSoup4, pypdf (for data extraction)
