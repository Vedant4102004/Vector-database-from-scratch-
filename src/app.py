import os
import sys
import time
import html
from pathlib import Path
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Ensure src/ directory is accessible for imports
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exact_index import ExactIndex
from ivf_index import IVFIndex

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vector Database From Scratch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Modern typography and card styling */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    background: linear-gradient(120deg, #1E88E5 0%, #7B1FA2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    font-size: 1.05rem;
    color: #616161;
    margin-bottom: 1.5rem;
}
.result-card {
    background-color: var(--secondary-background-color, #f8f9fa);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.result-card:hover {
    border-color: #1E88E5;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.rank-badge {
    background: #1E88E5;
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 2px 10px;
    border-radius: 12px;
}
.score-badge {
    background: rgba(30, 136, 229, 0.12);
    color: #1565C0;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(30, 136, 229, 0.25);
}
.id-badge {
    color: #757575;
    font-size: 0.8rem;
    font-family: monospace;
}
.match-badge {
    background: rgba(46, 125, 50, 0.15);
    color: #2E7D32;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(46, 125, 50, 0.3);
}
.card-text {
    font-size: 0.95rem;
    line-height: 1.45;
    color: var(--text-color, #212121);
}
.stats-banner {
    background: linear-gradient(90deg, rgba(30,136,229,0.06) 0%, rgba(123,31,162,0.06) 100%);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 0.9rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Database & Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_system():
    """
    Loads embedding model, precomputed dataset embeddings, texts,
    and initializes both ExactIndex and IVFIndex once.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name)

    embeddings_path = REPO_ROOT / "data" / "embeddings.npy"
    texts_path = REPO_ROOT / "data" / "texts.npy"

    if not embeddings_path.exists() or not texts_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {REPO_ROOT / 'data'}. "
            "Please run 'python src/load_data.py' and 'python src/generate_embeddings.py' first."
        )

    vectors = np.load(embeddings_path)
    texts = np.load(texts_path, allow_pickle=True)

    exact_index = ExactIndex(vectors)

    n_clusters = 500
    ivf_index = IVFIndex(vectors, n_clusters=n_clusters)
    ivf_index.build(iterations=10)

    return model, vectors, texts, exact_index, ivf_index


# Helper to compute IVF candidate count without changing IVFIndex class
def get_ivf_candidate_count(ivf_idx: IVFIndex, query_vec: np.ndarray, n_probe: int):
    """
    Inspects the IVF index centroids and inverted lists to determine
    the number of candidate vectors searched for this query.
    """
    q = query_vec.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    centroid_scores = ivf_idx.centroids @ q
    probe_clusters = np.argpartition(
        -centroid_scores,
        min(n_probe, ivf_idx.n_clusters) - 1
    )[:n_probe]
    candidate_count = sum(len(ivf_idx.inverted_lists.get(int(cid), [])) for cid in probe_clusters)
    return candidate_count, probe_clusters


# -----------------------------------------------------------------------------
# Application Startup
# -----------------------------------------------------------------------------
with st.spinner("⚡ Loading embedding model and building IVF index (500 clusters)..."):
    try:
        model, vectors, texts, exact_index, ivf_index = load_system()
        num_docs = len(vectors)
        embed_dim = vectors.shape[1]
    except Exception as e:
        st.error(f"Error loading vector database resources: {e}")
        st.stop()


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Search Controls")

    search_mode = st.radio(
        "Search Mode",
        options=["Exact Search", "Approximate Search", "Compare Both"],
        index=2,
        help="Exact: Brute force ground truth. Approximate: IVF cluster search. Compare: Side-by-side evaluation."
    )

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Number of nearest neighbours to retrieve."
    )

    if search_mode in ["Approximate Search", "Compare Both"]:
        n_probe = st.slider(
            "n_probe (Clusters to probe)",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Number of IVF clusters searched per query. Higher n_probe = higher recall, lower QPS."
        )
    else:
        n_probe = 10
        st.caption("ℹ️ *`n_probe` is only used for Approximate and Compare modes.*")

    st.markdown("---")
    st.markdown("### 📊 Index Telemetry")
    st.markdown(
        f"""
        - **Corpus**: AG News Dataset
        - **Total Vectors**: `{num_docs:,}`
        - **Dimensions**: `{embed_dim}`
        - **Embedding Model**: `all-MiniLM-L6-v2`
        - **Metric**: Cosine Similarity
        - **IVF Clusters**: `500`
        - **Index Source**: Built from scratch (NumPy)
        """
    )

    st.markdown("---")
    show_plots = st.checkbox("Show Benchmark Plots", value=False)


# -----------------------------------------------------------------------------
# Main Header & Query Input
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Vector Database From Scratch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive semantic search interface powered by custom NumPy Exact & IVF indexes (no FAISS/Pinecone).</div>',
    unsafe_allow_html=True
)

# Sample query presets covering different topics and recall characteristics
SAMPLE_QUERIES = [
    "technology earnings and stocks",
    "government elections and political campaign",
    "climate change and global warming agreement",
    "olympic games gold medal sprint winner",
]

# State for search query
if "search_input_box" not in st.session_state:
    st.session_state["search_input_box"] = SAMPLE_QUERIES[0]

st.markdown("**Try a sample statement:**")
cols = st.columns(len(SAMPLE_QUERIES))
for i, sample in enumerate(SAMPLE_QUERIES):
    if cols[i].button(sample[:28] + "...", key=f"sample_{i}", use_container_width=True):
        st.session_state["search_input_box"] = sample
        st.rerun()

query_text = st.text_input(
    "Enter a statement to search:",
    key="search_input_box",
    placeholder="Type any news statement, topic, or question...",
)


# -----------------------------------------------------------------------------
# Result Card Rendering Function
# -----------------------------------------------------------------------------
def render_results(results, ground_truth_ids=None):
    """Renders formatted cards for search results."""
    if not results:
        st.info("No matching results found.")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        is_exact_match = (ground_truth_ids is not None) and (doc_id in ground_truth_ids)
        exact_rank = ground_truth_ids.index(doc_id) + 1 if is_exact_match else None

        match_badge = (
            f'<span class="match-badge">🎯 Ground Truth #{exact_rank}</span>&nbsp;'
            if is_exact_match
            else ""
        )

        raw_text = texts[doc_id] if doc_id < len(texts) else "Unknown Document"
        safe_text = html.escape(str(raw_text))

        card_html = (
            f'<div class="result-card">'
            f'<div class="result-header">'
            f'<div>'
            f'<span class="rank-badge">Rank #{rank}</span> &nbsp;'
            f'<span class="id-badge">Doc ID: {doc_id}</span>'
            f'</div>'
            f'<div>'
            f'{match_badge}'
            f'<span class="score-badge">Similarity: {score:.4f}</span>'
            f'</div>'
            f'</div>'
            f'<div class="card-text">{safe_text}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Search Execution
# -----------------------------------------------------------------------------
if query_text.strip():
    # Encode query vector using sentence-transformers
    query_vector = model.encode(query_text.strip())

    # Mode 1: Exact Search Only
    if search_mode == "Exact Search":
        t0 = time.perf_counter()
        results = exact_index.search(query_vector, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        qps = 1000.0 / latency_ms if latency_ms > 0 else 0

        st.markdown("---")
        st.markdown(
            f'<div class="stats-banner">🎯 <b>Exact Brute-Force Search:</b> Retrieved Top-{top_k} results across all <b>{num_docs:,}</b> vectors in <b>{latency_ms:.2f} ms</b> ({qps:.1f} QPS).</div>',
            unsafe_allow_html=True
        )
        render_results(results)

    # Mode 2: Approximate Search Only
    elif search_mode == "Approximate Search":
        # Compute candidates searched without modifying core IVF class
        candidate_count, probe_clusters = get_ivf_candidate_count(ivf_index, query_vector, n_probe)
        candidate_pct = (candidate_count / num_docs) * 100

        t0 = time.perf_counter()
        results = ivf_index.search(query_vector, top_k=top_k, n_probe=n_probe)
        latency_ms = (time.perf_counter() - t0) * 1000
        qps = 1000.0 / latency_ms if latency_ms > 0 else 0

        st.markdown("---")
        st.markdown(
            f'<div class="stats-banner">⚡ <b>IVF Approximate Search:</b> Searched <b>{candidate_count:,}</b> candidates ({candidate_pct:.2f}% of dataset) across <b>{n_probe}</b> probed clusters in <b>{latency_ms:.2f} ms</b> ({qps:.1f} QPS).</div>',
            unsafe_allow_html=True
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Latency", f"{latency_ms:.2f} ms")
        col_m2.metric("Throughput", f"{qps:.1f} QPS")
        col_m3.metric("Candidates Searched", f"{candidate_count:,}", f"{candidate_pct:.1f}% of index")
        col_m4.metric("Clusters Probed", f"{n_probe} / 500", f"{(n_probe / 500) * 100:.1f}%")

        st.markdown("### Top Results")
        render_results(results)

    # Mode 3: Compare Both
    elif search_mode == "Compare Both":
        # Candidate count calculation for IVF
        candidate_count, probe_clusters = get_ivf_candidate_count(ivf_index, query_vector, n_probe)
        candidate_pct = (candidate_count / num_docs) * 100

        # Run Exact search
        t_exact_0 = time.perf_counter()
        exact_results = exact_index.search(query_vector, top_k=top_k)
        exact_time_ms = (time.perf_counter() - t_exact_0) * 1000
        exact_qps = 1000.0 / exact_time_ms if exact_time_ms > 0 else 0

        # Run IVF Approximate search
        t_ivf_0 = time.perf_counter()
        ivf_results = ivf_index.search(query_vector, top_k=top_k, n_probe=n_probe)
        ivf_time_ms = (time.perf_counter() - t_ivf_0) * 1000
        ivf_qps = 1000.0 / ivf_time_ms if ivf_time_ms > 0 else 0

        # Recall@K and Overlap calculation
        exact_top_k_ids = [doc_id for doc_id, _ in exact_results]
        approx_top_k_ids = [doc_id for doc_id, _ in ivf_results]

        exact_ids = set(exact_top_k_ids)
        approx_ids = set(approx_top_k_ids)

        matches = len(exact_ids.intersection(approx_ids))
        recall_at_k = matches / top_k if top_k > 0 else 0.0
        speedup = (exact_time_ms / ivf_time_ms) if ivf_time_ms > 0 else 1.0

        st.markdown("---")
        st.markdown("### 🏆 Side-by-Side Benchmark & Metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"Recall@{top_k}",
            f"{recall_at_k * 100:.1f}%",
            f"{matches}/{top_k} Ground Truth Matches"
        )
        m2.metric(
            "Approximate Latency (IVF)",
            f"{ivf_time_ms:.2f} ms",
            f"{ivf_qps:.1f} QPS"
        )
        m3.metric(
            "Exact Latency (Brute Force)",
            f"{exact_time_ms:.2f} ms",
            f"{exact_qps:.1f} QPS"
        )
        m4.metric(
            "Candidates Evaluated",
            f"{candidate_count:,}",
            f"{candidate_pct:.1f}% of full index"
        )

        st.markdown(
            f'<div class="stats-banner">💡 <b>Tradeoff Insight:</b> At <code>n_probe={n_probe}</code>, IVF examined only <b>{candidate_count:,}</b> vectors ({candidate_pct:.2f}% of corpus), achieving <b>{recall_at_k * 100:.1f}% Recall@{top_k}</b> ({matches}/{top_k} exact matches).</div>',
            unsafe_allow_html=True
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"#### 🎯 Exact Search (Ground Truth)")
            st.caption(f"Brute-force scan of {num_docs:,} vectors • {exact_time_ms:.2f} ms")
            render_results(exact_results)

        with col_right:
            st.markdown(f"#### ⚡ Approximate Search (IVF)")
            st.caption(f"{n_probe} clusters probed • {candidate_count:,} candidates • {ivf_time_ms:.2f} ms")
            render_results(ivf_results, ground_truth_ids=exact_top_k_ids)

else:
    st.warning("Please enter a query statement above.")


# -----------------------------------------------------------------------------
# Benchmark Visualizations (Optional View)
# -----------------------------------------------------------------------------
if show_plots:
    st.markdown("---")
    st.markdown("### 📈 Precomputed Benchmark Results")
    plot_dir = REPO_ROOT / "plots"
    p1 = plot_dir / "recall_vs_nprobe.png"
    p2 = plot_dir / "speed_vs_nprobe.png"
    p3 = plot_dir / "recall_vs_clusters.png"
    p4 = plot_dir / "speed_vs_clusters.png"

    plot_cols = st.columns(2)
    if p1.exists():
        plot_cols[0].image(str(p1), caption="IVF Recall@10 vs n_probe", use_container_width=True)
    if p2.exists():
        plot_cols[1].image(str(p2), caption="IVF Speed (QPS) vs n_probe", use_container_width=True)

    plot_cols2 = st.columns(2)
    if p3.exists():
        plot_cols2[0].image(str(p3), caption="IVF Recall vs Number of Clusters", use_container_width=True)
    if p4.exists():
        plot_cols2[1].image(str(p4), caption="IVF Speed vs Number of Clusters", use_container_width=True)


# -----------------------------------------------------------------------------
# How It Works Section
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📚 How It Works: Under the Hood of This Vector Database", expanded=False):
    st.markdown(
        """
        ### 1. Dense Text Embeddings
        Each AG News article is converted into a **384-dimensional dense vector** using the pre-trained 
        `sentence-transformers/all-MiniLM-L6-v2` model. Sentences with similar semantic concepts 
        (e.g., *stocks*, *quarterly earnings*, *Wall Street*) are mapped close together in this high-dimensional vector space.

        ### 2. Cosine Similarity via Dot Product
        All 50,000 vectors and incoming query vectors are normalized to unit length ($L_2$-norm = 1):
        $$\\hat{\\mathbf{u}} = \\frac{\\mathbf{u}}{\\|\\mathbf{u}\\|_2}$$
        Because vectors are normalized, cosine similarity reduces to a fast matrix-vector dot product:
        $$\\text{cosine\\_similarity}(\\mathbf{u}, \\mathbf{v}) = \\hat{\\mathbf{u}} \\cdot \\hat{\\mathbf{v}}$$
        Scores range from `-1.0` (opposite) to `+1.0` (identical).

        ### 3. Exact Nearest Neighbour Index (`ExactIndex`)
        - **Mechanism:** Brute-force matrix multiplication: $\\mathbf{S} = \\mathbf{V} \\cdot \\mathbf{q}$.
        - **Computational Complexity:** $\\mathcal{O}(N \\cdot d)$, where $N = 50,000$ and $d = 384$.
        - **Accuracy:** **100% Ground Truth** recall guaranteed.
        - **Limitation:** Throughput drops linearly as dataset size $N$ scales into millions.

        ### 4. IVF (Inverted File) Approximate Index (`IVFIndex`)
        - **Partitioning:** The vector space is divided into **500 Voronoi cells (clusters)** using k-means++ centroid initialization.
        - **Inverted Lists:** Vectors are mapped to their nearest centroid. The index stores a dictionary mapping each cluster ID to its list of document IDs.
        - **Querying:** When a query arrives, it is first compared against the **500 cluster centroids** to find the closest clusters.

        ### 5. The `n_probe` Tuning Knob
        - **Accuracy vs. Speed Dial:** `n_probe` dictates how many of the closest clusters to inspect.
        - **Low `n_probe` (1 - 5):** Evaluates only ~1% of vectors $\\rightarrow$ extremely high QPS, lower recall.
        - **High `n_probe` (25 - 50):** Evaluates more clusters $\\rightarrow$ recall reaches **>98%** while still skipping most of the dataset!
        """
    )
