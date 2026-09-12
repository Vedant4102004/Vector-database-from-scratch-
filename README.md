# Vector Database From Scratch

A vector database implemented from scratch using NumPy, without using Pinecone, FAISS, Chroma, or sklearn.neighbors.

## Problem Statement

The goal is to understand and implement the core ideas behind vector databases and approximate nearest-neighbour search.

The system supports:

- Exact nearest-neighbour search
- Approximate nearest-neighbour search using IVF
- Configurable speed vs accuracy using `n_probe`
- Insert operations
- Delete operations
- Benchmarking against exact ground truth
- Recall and queries-per-second evaluation

## Architecture

```text
AG News
   |
   v
all-MiniLM-L6-v2
   |
   v
50,000 text embeddings
384 dimensions
   |
   +----------------------+
   |                      |
   v                      v
Exact Index             IVF Index
Brute-force             500 clusters
cosine search            |
   |                      v
   |                   n_probe
   |                      |
   +----------+-----------+
              |
              v
       Top-k nearest neighbours
