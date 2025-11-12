# app/streamlit_app.py
"""
RAG Hallucination Detection — TruthfulQA + Academic Paper Verification
Author: Sai Srivastav
"""

import os
import random
import re
import warnings
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support

# Local modules
from academic_retriever import SemanticScholar, credibility_score
from rag_core import Generator, Retriever, Verifier

# or "none" if needed while debugging
# fileWatcherType = "none"


# -----------------------------------------------------------------------------
# Environment & Warnings
# -----------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------------------------
st.set_page_config(

    page_title="RAG Hallucination — TruthfulQA + Academic Paper Check",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Cached Resource Loaders (parameterized so Streamlit caches per setting)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_generator(use_gemini: bool, model_name: str):
    """Initialize text generator (Gemini or local)."""
    return Generator(use_gemini=use_gemini, model_name=model_name)

@st.cache_resource(show_spinner=False)
def load_retriever(use_gemini: bool, model_name: str, sleep: float):
    """Retriever that prefers Gemini factual context; falls back to Wikipedia."""
    return Retriever(use_gemini=use_gemini, model_name=model_name, sleep=sleep)

@st.cache_resource(show_spinner=False)
def load_verifier(threshold: float):
    """Verifier for factual consistency (single threshold)."""
    return Verifier(threshold=threshold)

@st.cache_resource(show_spinner=False)
def load_embedder():
    """Shared embedding model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def load_truthful_qa():
    """Load TruthfulQA generation split."""
    ds = load_dataset("truthful_qa", "generation")
    return list(ds["validation"])

@st.cache_resource(show_spinner=False)
def load_semantic_scholar():
    """Semantic Scholar API wrapper."""
    return SemanticScholar()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sim_ref(a: str, r: str, emb: SentenceTransformer) -> float:
    """Cosine similarity between answer and reference."""
    if not a or not r:
        return 0.0
    va = emb.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    vr = emb.encode(r, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(va, vr)[0][0].item())

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

use_gemini = st.sidebar.checkbox(
    "Use Google Gemini", value=bool(os.getenv("GEMINI_API_KEY")), key="use_gemini"
)

model_name = st.sidebar.selectbox(
    "Model",
    ["gemini-2.5-flash", "gemini-2.5-pro", "google/flan-t5-large"],
    index=0 if use_gemini else 2,
    key="model_name",
)

threshold = st.sidebar.slider(
    "Verdict threshold (similarity)", 0.0, 1.0, 0.72, 0.01, key="threshold"
)

limit = st.sidebar.number_input(
    "Batch size (TruthfulQA)", min_value=10, max_value=817, value=50, step=10, key="limit"
)
seed = st.sidebar.number_input("Random seed", min_value=1, value=42, step=1, key="seed")

if use_gemini:
    if not os.getenv("GEMINI_API_KEY"):
        st.sidebar.warning("⚠️ GEMINI_API_KEY not found — using local model instead.")
    else:
        st.sidebar.caption("✅ Using Gemini model (API key found).")
else:
    st.sidebar.caption("Using local fallback model (Flan-T5)")

# -----------------------------------------------------------------------------
# Load Core Components (respect sidebar settings)
# -----------------------------------------------------------------------------
effective_use_gemini = use_gemini and "gemini" in model_name
gen = load_generator(effective_use_gemini, model_name)
ret = load_retriever(effective_use_gemini, model_name, sleep=0.1)
ver = load_verifier(threshold)
embedder = load_embedder()
data = load_truthful_qa()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Hallucination Detection using RAG — TruthfulQA + Academic Paper Check")
st.caption(f"Generator: {gen.kind} | Threshold: {threshold:.2f}")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(["🔎 Live Demo", "📦 Batch Evaluate", "📊 Charts", "📚 Paper Check"])

# -----------------------------------------------------------------------------
# TAB 1 — Live Demo
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Live Demo (single question)")
    q = st.text_input("Enter a question (e.g., 'Who discovered penicillin?')", key="live_q")
    if st.button("Run Live Demo", key="btn_run_live"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating, retrieving, and verifying..."):
                ctx = ret.retrieve(q)
                evidence = ctx.get("context", "")

                prompt = f"""Answer truthfully using only the context below.

Question: {q}

Context:
{evidence}

Answer in 2–4 sentences:
"""
                ans = gen.generate(prompt)
                verdict, score = ver.verdict(ans, evidence)

            st.markdown("**Model Output:**")
            st.write(ans or "_empty response_")
            st.markdown("**Retrieved Context:**")
            st.code(evidence or "(no context)")
            st.caption(f"Sources: {', '.join(ctx.get('sources', [])) or '(none)'}")
            st.metric("Factual Consistency Score", f"{score:.3f}")

            if verdict == "Factual":
                st.success("✅ Factual")
            elif verdict == "Uncertain":
                st.warning("⚠️ Uncertain — partial support")
            else:
                st.error("❌ Hallucinated")

# -----------------------------------------------------------------------------
# TAB 2 — Batch Evaluate (TruthfulQA)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Batch Evaluation on TruthfulQA")
    st.caption("Evaluate a random subset and compute factual metrics.")
    colA, colB = st.columns([1, 1])
    run_batch = colA.button("Run Batch Now", key="btn_batch")
    show_last = colB.button("Show Last Batch", key="btn_show_last")

    if run_batch:
        random.seed(seed)
        subset = random.sample(data, k=min(limit, len(data)))
        rows = []
        prog = st.progress(0)

        for i, ex in enumerate(subset, start=1):
            q = ex["question"]
            ref = ex.get("best_answer", "")
            ans = gen.generate(q)
            ctx = ret.retrieve(q)
            verdict, score = ver.verdict(ans, ctx.get("context", ""))
            rows.append(
                dict(
                    idx=i,
                    question=q,
                    truthful_ref=ref,
                    model_output=ans,
                    context=ctx.get("context", ""),
                    sources="; ".join(ctx.get("sources", [])),
                    score=round(score, 4),
                    verdict=verdict,
                )
            )
            prog.progress(i / len(subset))

        st.session_state["last_df"] = pd.DataFrame(rows)
        st.success(f"Batch complete — {len(rows)} samples")
        st.dataframe(st.session_state["last_df"].head(20), use_container_width=True)

    if show_last and "last_df" in st.session_state:
        st.dataframe(st.session_state["last_df"].head(50), use_container_width=True)

    if "last_df" in st.session_state:
        df = st.session_state["last_df"]

        # Weak metrics using similarity to reference answer
        df["score_ref"] = df.apply(
            lambda x: sim_ref(x["model_output"], x["truthful_ref"], embedder),
            axis=1,
        )
        gt = (df["score_ref"] >= threshold).astype(int)  # weak ground truth
        pred = (df["verdict"] == "Factual").astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        st.markdown(
            f"**Precision:** {p:.3f} | **Recall:** {r:.3f} | **F1:** {f1:.3f} (vs reference)"
        )
        st.bar_chart(df["verdict"].value_counts())

# -----------------------------------------------------------------------------
# TAB 3 — Charts
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Charts")
    if "last_df" not in st.session_state:
        st.info("Run a batch first in the previous tab.")
    else:
        df = st.session_state["last_df"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["score"].astype(float), bins=30)
        ax.set_title("Distribution of Factual Consistency Scores")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Confusion breakdown vs ref
        if "score_ref" not in df.columns:
            df["score_ref"] = df.apply(
                lambda x: sim_ref(x["model_output"], x["truthful_ref"], embedder),
                axis=1,
            )
        gt = (df["score_ref"] >= threshold).astype(int)
        pred = (df["verdict"] == "Factual").astype(int)

        tn = int(((gt == 0) & (pred == 0)).sum())
        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        st.write(f"TN: {tn} | TP: {tp} | FP: {fp} | FN: {fn}")

# -----------------------------------------------------------------------------
# TAB 4 — Academic Paper Check
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Academic Paper Check")
    st.caption("Search Semantic Scholar and verify claim support using abstracts.")

    q_col, n_col = st.columns([3, 1])
    q_paper = q_col.text_input("Enter a research question", key="paper_q")
    topn = n_col.number_input("Top N Papers", 1, 20, 5, 1, key="paper_topn")

    if st.button("Search & Verify", key="btn_search_papers"):
        s2 = load_semantic_scholar()

        with st.spinner("Searching Semantic Scholar..."):
            papers = s2.search(q_paper.strip(), limit=int(topn))

        if not papers:
            st.warning("No papers found.")
        else:
            st.success(f"Found {len(papers)} papers.")
            rows = []
            q_emb = embedder.encode(q_paper, convert_to_tensor=True, normalize_embeddings=True)

            for p in papers:
                title = p.get("title", "(no title)")
                abs_ = p.get("abstract", "")
                url = p.get("url", "")
                year = p.get("year", "")
                venue = p.get("venue", "")
                cites = int(p.get("citationCount") or 0)

                sim = 0.0
                if abs_:
                    a_emb = embedder.encode(abs_, convert_to_tensor=True, normalize_embeddings=True)
                    sim = float(util.cos_sim(q_emb, a_emb)[0][0].item())

                cred = credibility_score(p, query_similarity=sim)

                with st.expander(
                    f"📄 {title} ({year}) — {venue} | cites: {cites} | score: {cred:.2f}"
                ):
                    st.write(abs_ or "_No abstract available_")
                    if url:
                        st.markdown(f"[Open on Semantic Scholar]({url})")

                rows.append(
                    dict(
                        title=title,
                        year=year,
                        venue=venue,
                        citations=cites,
                        sim_query_abs=round(sim, 3),
                        credibility=cred,
                        url=url,
                        abstract=abs_,
                    )
                )

            df = pd.DataFrame(rows).sort_values(
                ["credibility", "sim_query_abs", "citations"], ascending=False
            )

            st.dataframe(
                df[["title", "year", "venue", "citations", "sim_query_abs", "credibility", "url"]],
                use_container_width=True,
            )

            # Build evidence from top abstracts
            top_ctx = [
                f"{r['title']} ({r['year']}): {r['abstract']}"
                for _, r in df.head(3).iterrows()
                if r["abstract"]
            ]
            evidence = "\n\n".join(top_ctx)[:2000]

            st.markdown("### Academic Answer (Grounded on Top Abstracts)")
            prompt = f"""You are an academic assistant.
Answer the question ONLY using the evidence below.
Use a neutral tone and include inline citations like [1], [2].

Question: {q_paper}

Evidence:
{evidence}

Answer:"""
            ans = gen.generate(prompt)
            st.write(ans)

            # Reuse global threshold for consistency
            ver_local = Verifier(threshold=threshold)
            verdict, score = ver_local.verdict(ans, evidence)
            st.metric("Factual Consistency Score (vs abstracts)", f"{score:.3f}")

            if verdict == "Factual":
                st.success("✅ Academically Supported")
            elif verdict == "Uncertain":
                st.warning("⚠️ Uncertain — partial support")
            else:
                st.error("❌ Not Supported / Hallucinated")

            st.markdown("### References")
            for i, (_, r) in enumerate(df.head(3).iterrows(), start=1):
                ref = f"[{i}] {r['title']} ({r['year']}). {r['venue']}. cites={r['citations']}"
                if r["url"]:
                    ref += f" — [link]({r['url']})"
                st.markdown(ref)
