import json
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from app.rag_core import Generator, Retriever, Verifier
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from tqdm import tqdm


def sim_ref(a, r, enc):
    if not a or not r:
        return 0.0
    va = enc.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    vr = enc.encode(r, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(va, vr)[0][0].item())


def main(limit=817, threshold=0.72, use_openai=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    gen = Generator(use_openai=use_openai)  # GPT-3.5 if key set, else flan-t5-small
    ret = Retriever(k_titles=4, k_sents=8, sleep=0.1)
    ver = Verifier(threshold=threshold)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Generator: {gen.kind} | Threshold: {threshold} | Limit: {limit}")

    ds = load_dataset("truthful_qa", "generation")["validation"]
    rows = list(ds)
    if limit < len(rows):
        rows = random.sample(rows, limit)

    recs = []
    for i, ex in enumerate(tqdm(rows, desc="Evaluating")):
        q = ex["question"]
        ref = ex.get("best_answer", "")
        ans = gen.generate(q)
        ctx = ret.retrieve(q)
        verdict, score = ver.verdict(ans, ctx["context"])
        s_ref = sim_ref(ans, ref, enc)
        recs.append(
            {
                "idx": i + 1,
                "question": q,
                "truthful_ref": ref,
                "model_output": ans,
                "context": ctx["context"],
                "sources": "; ".join(ctx["sources"]),
                "score_evidence": round(score, 4),
                "score_ref": round(s_ref, 4),
                "verdict": verdict,
            }
        )

    df = pd.DataFrame(recs)
    df.to_csv("truthfulqa_full_results.csv", index=False)
    print("Saved: truthfulqa_full_results.csv")

    gt = (df["score_ref"].astype(float) >= threshold).astype(int)
    pred = (df["verdict"] == "Factual").astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(gt, df["score_evidence"].astype(float))
    except Exception:
        auc = float("nan")

    metrics = {
        "n": len(df),
        "generator": gen.kind,
        "threshold": threshold,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "auc_score_evidence": round(auc, 4) if not np.isnan(auc) else None,
        "verdict_counts": dict(Counter(df["verdict"])),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plt.figure(figsize=(7, 4.5))
    plt.hist(df["score_evidence"].astype(float), bins=30)
    plt.title("Distribution of Factual Consistency Scores")
    plt.xlabel("Score (answer vs evidence)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("score_hist.png", dpi=300)

    cm = confusion_matrix(gt, pred, labels=[0, 1])
    fig = plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (0=Non-factual,1=Factual)")
    for (x, y), v in np.ndenumerate(cm):
        plt.text(y, x, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["GT 0", "GT 1"])
    plt.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=300)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=817)
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--use_openai", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(
        limit=args.limit,
        threshold=args.threshold,
        use_openai=args.use_openai,
        seed=args.seed,
    )
