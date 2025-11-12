"""
rag_core.py
Core components for RAG-based hallucination detection:
    - Generator: text generation (Gemini or local model)
    - Retriever: factual context retrieval (Gemini or Wikipedia)
    - Verifier: semantic consistency scoring
"""

import os
import time
import warnings
from typing import Dict
# at top of file
import google.generativeai as genai


import torch
from sentence_transformers import SentenceTransformer, util

# Silence spurious torch warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")


def _select_device_str() -> str:
    """Return a device string usable by HF pipelines and SentenceTransformers."""
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------- Generator -------------------------
class Generator:
    """Unified text generator — uses Gemini if available, else FLAN-T5."""

    def __init__(self, use_gemini: bool = False, model_name: str = "gemini-2.5-flash"):
        self.kind = "flan-t5-large"
        self.use_gemini = use_gemini and bool(os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self._genai = None

        if self.use_gemini:
            try:
                import google.generativeai as genai  # lazy import
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._genai = genai
                self.model = genai.GenerativeModel(self.model_name)
                self.kind = f"gemini:{self.model_name}"
            except Exception as e:
                print(f"[Generator] Gemini init failed ({e}); using FLAN fallback.")
                self.use_gemini = False

        if not self.use_gemini:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
            mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            self.pipe = pipeline("text2text-generation", model=mod, tokenizer=tok)

    def generate(
        self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2
    ) -> str:
        """Generate a text completion for the given prompt."""
        if not prompt.strip():
            return ""

        # Prefer Gemini when active
        if self.use_gemini and self._genai:
            try:
                out = self.model.generate_content(prompt)
                return (getattr(out, "text", None) or "").strip()
            except Exception as e:
                return f"[Gemini Error] {e}"

        # Fallback: local transformer
        try:
            out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0][
                "generated_text"
            ]
            return out.strip()
        except Exception as e:
            return f"[Local model error] {e}"




# ------------------------- Retriever -------------------------
class Retriever:
    """Retrieves factual background context from Gemini or Wikipedia."""

    def __init__(
        self,
        use_gemini: bool = True,
        model_name: str = "gemini-2.5-flash",
        sleep: float = 0.15,
    ):
        self.use_gemini = use_gemini and bool(os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.sleep = sleep
        self._genai = None
        self.model = None
        self.wikipedia = None

        if self.use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._genai = genai
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"[Retriever] Gemini init failed ({e}); falling back to Wikipedia.")
                self.use_gemini = False

        if not self.use_gemini:
            import wikipedia  # lazy import only when needed

            self.wikipedia = wikipedia

    def retrieve(self, query: str, max_chars: int = 1600) -> Dict[str, str]:
        """Retrieve concise factual context for a query."""
        query = (query or "").strip()
        if not query:
            return {"context": "", "sources": []}

        # Gemini factual context
        if self.use_gemini and self._genai and self.model:
            try:
                prompt = (
                    "Provide a factual, concise background for this question.\n\n"
                    f"Question: {query}\n\n"
                    "Use verified, general knowledge in 4–6 sentences. Avoid opinions."
                )
                resp = self.model.generate_content(prompt)
                ctx = (getattr(resp, "text", None) or "").strip()
                time.sleep(self.sleep)
                return {
                    "context": ctx[:max_chars],
                    "sources": ["Gemini factual context"],
                }
            except Exception as e:
                print(f"[Retriever: Gemini Error] {e}")
                # continue to Wikipedia fallback below

        # Wikipedia fallback
        try:
            results = self.wikipedia.search(query, results=3) if self.wikipedia else []
            summaries = []
            for title in results:
                try:
                    summaries.append(self.wikipedia.summary(title, sentences=3))
                except Exception:
                    continue
            text = " ".join(summaries).strip()[:max_chars]
            return {"context": text, "sources": results[:3] if results else []}
        except Exception as e:
            print(f"[Retriever: Wikipedia Error] {e}")
            return {"context": "", "sources": []}


# ------------------------- Verifier -------------------------
# ----------------- VERIFIER -----------------
from sentence_transformers import SentenceTransformer, util

class Verifier:
    """Checks factual consistency between model output and retrieved context."""

    def __init__(self, threshold: float = 0.72):
        self.threshold = float(threshold)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def verdict(self, answer: str, context: str):
        """Return (verdict, score) based on cosine similarity."""
        answer = (answer or "").strip()
        context = (context or "").strip()
        if not answer or not context:
            return "Hallucinated", 0.0

        a = self.model.encode(answer, convert_to_tensor=True, normalize_embeddings=True)
        b = self.model.encode(context, convert_to_tensor=True, normalize_embeddings=True)
        score = float(util.cos_sim(a, b)[0][0].item())

        verdict = "Factual" if score >= self.threshold else "Hallucinated"
        return verdict, score

