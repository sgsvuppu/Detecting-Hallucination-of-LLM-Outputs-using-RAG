"""
gpt_precision.py — Optional GPT-based precision layer for RAG.
Adds:
  1. Query Expansion  (for better retrieval)
  2. Context Reranking (prioritize relevant sentences)
  3. Answer Refinement (rewrite model output using evidence)
  4. Factual Judge (verify factual alignment)
"""

import os

from openai import OpenAI

# Initialize OpenAI client (make sure your key is set)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------- 1️⃣ Query Expansion ----------
def expand_queries(question: str):
    """
    Uses GPT to generate alternative phrasings or sub-queries.
    """
    prompt = f"""
You are a retrieval optimization model.
Given a user query, generate 3 semantically different sub-queries that could
help retrieve better evidence. Respond with each on a new line.

Query: "{question}"
Sub-queries:
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        queries = [q.strip("- ").strip() for q in lines if q.strip()]
        return [question] + queries[:3]
    except Exception:
        return [question]


# ---------- 2️⃣ Reranking ----------
def gpt_rerank(query: str, sentences, top_n: int = 5):
    """
    Rerank retrieved sentences by relevance to the query using GPT.
    """
    if not sentences:
        return []
    snippet_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences[:20])])
    prompt = f"""
Rank the following text snippets by how relevant they are to the question.

Question: "{query}"

Snippets:
{snippet_text}

Return the indices of the {top_n} most relevant snippets in descending order of relevance.
Format: comma-separated numbers only.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        text = resp.choices[0].message.content.strip()
        nums = [int(x) for x in text.replace(" ", "").split(",") if x.isdigit()]
        chosen = [sentences[i - 1] for i in nums if 0 < i <= len(sentences)]
        return chosen[:top_n]
    except Exception:
        return sentences[:top_n]


# ---------- 3️⃣ Refinement ----------
def refine_with_context(answer: str, context: str):
    """
    Refines or rewrites the generated answer using retrieved context.
    """
    prompt = f"""
You are a factuality assistant.
Given the following context and an answer, rewrite the answer to be
factually consistent with the context. Keep it concise and accurate.

Context:
{context}

Answer:
{answer}

Refined factual answer:
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return answer


# ---------- 4️⃣ Factual Judge ----------
def judge_support(question: str, answer: str, context: str):
    """
    Checks whether the answer is actually supported by the evidence.
    """
    prompt = f"""
You are a factual judge.
Given a question, an answer, and context evidence, determine whether
the answer is supported by the context.

Question: {question}
Answer: {answer}
Context: {context}

Respond in JSON:
{{"supported": true/false, "explanation": "short reason"}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()

        # Naive JSON-safe parse
        if '"supported": true' in text.lower():
            return {"supported": True, "raw": text}
        elif '"supported": false' in text.lower():
            return {"supported": False, "raw": text}
        else:
            return {"supported": None, "raw": text}
    except Exception as e:
        return {"supported": None, "error": str(e)}
