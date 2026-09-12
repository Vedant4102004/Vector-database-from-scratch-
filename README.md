````markdown
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
````

## Dataset

The project uses the AG News dataset.

* 50,000 text samples
* 384-dimensional embeddings
* Embedding model: `all-MiniLM-L6-v2`

The generated embeddings are intentionally excluded from Git because of their large size.

## Exact Index

The exact index performs brute-force cosine similarity search over all vectors.

For a query vector `q` and vector `x`:

```text
cosine_similarity(q, x) = q · x
```

after L2 normalization.

The exact index provides the ground truth used to evaluate the approximate index.

## IVF Approximate Index

The approximate index uses an Inverted File (IVF) structure.

### Index construction

1. Initialize 500 centroids.
2. Assign vectors to their closest centroid.
3. Recalculate centroids.
4. Repeat the clustering process.
5. Store vector IDs in inverted lists corresponding to their clusters.

### Search

Instead of comparing the query against all 50,000 vectors:

1. Compare the query with all centroids.
2. Select the closest `n_probe` clusters.
3. Search only vectors inside those clusters.
4. Return the top-k results.

`n_probe` controls the speed/accuracy trade-off.

Lower `n_probe`:

* Searches fewer vectors
* Faster
* Lower recall

Higher `n_probe`:

* Searches more vectors
* Slower
* Higher recall

## Benchmark

The benchmark uses:

* 500 query vectors
* Top-10 nearest neighbours
* Exact brute-force search as ground truth
* Recall@10 as the accuracy metric

Example benchmark:

| n_probe | Recall@10 | Queries/sec |
| ------: | --------: | ----------: |
|       1 |    0.7482 |     4526.96 |
|       5 |    0.9236 |     1961.93 |
|      10 |    0.9522 |      827.13 |
|      25 |    0.9782 |      309.01 |
|      50 |    0.9892 |      155.58 |

Exact search in the benchmark run achieved approximately:

```text
144.85 queries/sec
```

The exact QPS can vary depending on system load.

The results demonstrate the expected IVF trade-off: increasing `n_probe` improves recall while reducing search throughput.

## Plots

The benchmark automatically generates:

```text
plots/recall_vs_nprobe.png
plots/speed_vs_nprobe.png
```

These files are generated locally and are excluded from Git.

## Insert

New vectors can be inserted without rebuilding the entire IVF index.

The vector is:

1. Normalized.
2. Assigned to its nearest centroid.
3. Added to that centroid's inverted list.

## Delete

Vectors can be removed from their corresponding inverted list using their vector ID.

## Project Structure

```text
VectorDB_from_scratch/
│
├── data/
│   └── embeddings.npy
│
├── plots/
│   ├── recall_vs_nprobe.png
│   └── speed_vs_nprobe.png
│
├── src/
│   ├── benchmark.py
│   ├── exact_index.py
│   ├── generate_embeddings.py
│   ├── ivf_index.py
│   ├── load_data.py
│   ├── test_exact.py
│   ├── test_ivf.py
│   └── test_operations.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate the Dataset and Embeddings

Load the AG News dataset:

```bash
python src/load_data.py
```

Generate embeddings:

```bash
python src/generate_embeddings.py
```

This creates the local embedding files inside:

```text
data/
```

## Run Exact Search

```bash
python src/test_exact.py
```

## Build and Test IVF

```bash
python src/test_ivf.py
```

## Run Insert/Delete Test

```bash
python src/test_operations.py
```

## Run Benchmark

```bash
python src/benchmark.py
```

This computes exact ground truth, evaluates IVF at multiple `n_probe` settings, and generates the benchmark plots.

## Technologies

* Python
* NumPy
* Pandas
* Hugging Face Datasets
* Sentence Transformers
* Matplotlib

## Restrictions Followed

This implementation does not use:

* Pinecone
* FAISS
* Chroma
* sklearn.neighbors
* Any external vector database

The vector indexing and search logic is implemented directly using NumPy.

## Limitations

This is an educational implementation designed to demonstrate the internal concepts behind vector search.

The current IVF implementation does not include production database features such as:

* Persistent index serialization
* Distributed indexing
* Concurrent writes
* Advanced centroid initialization
* Background index rebuilding
* Memory-mapped storage

## Key Result

The project demonstrates that IVF can significantly improve search throughput compared with brute-force search while providing a configurable recall/speed trade-off through `n_probe`.

For example, with `n_probe=1`, the implementation achieved approximately 4,527 queries/sec with 74.82% Recall@10, while `n_probe=50` achieved approximately 99% Recall@10 at approximately 156 queries/sec.

```

Save:

- `Ctrl + O`
- Enter
- `Ctrl + X`

Then **don't commit yet**.

Reply `done` after saving it.
```
