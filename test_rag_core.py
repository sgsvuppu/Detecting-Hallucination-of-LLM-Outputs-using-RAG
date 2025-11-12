import types
from unittest.mock import MagicMock, patch

import pytest
import app.rag_core as rc


# ---------- Fixtures ----------
@pytest.fixture
def fake_sentence_transformer():
    class FakeST:
        def encode(self, text, convert_to_tensor=True, normalize_embeddings=True):
            # represent string as simple vector of length = len(text)
            # return a list-like with deterministic scalar for simplicity
            return [float(len(text or ""))]

    return FakeST()


def fake_cos_sim(a, b):
    # cosine similarity for scalars (non-negative) = 1 if both > 0 else 0
    av = a[0] if isinstance(a, (list, tuple)) else a
    bv = b[0] if isinstance(b, (list, tuple)) else b
    if av == 0.0 or bv == 0.0:
        return [[types.SimpleNamespace(item=lambda: 0.0)]]
    return [[types.SimpleNamespace(item=lambda: min(av, bv) / max(av, bv))]]


# ---------- Generator tests ----------
@patch("rag_core.genai")
@patch("rag_core.AutoTokenizer")
@patch("rag_core.AutoModelForSeq2SeqLM")
@patch("rag_core.pipeline")
def test_generator_uses_gemini_when_key_present(
    mock_pipe, mock_model, mock_tok, mock_genai, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    mock_resp = types.SimpleNamespace(text=" Hello ")
    mock_model_inst = types.SimpleNamespace(generate_content=lambda p: mock_resp)
    mock_genai.GenerativeModel.return_value = mock_model_inst

    g = rc.Generator(use_gemini=True)
    out = g.generate("prompt")

    assert out == "Hello"
    mock_genai.configure.assert_called()
    mock_genai.GenerativeModel.assert_called_with(g.model_name)
    mock_pipe.assert_not_called()


@patch("rag_core.genai")
@patch("rag_core.AutoTokenizer")
@patch("rag_core.AutoModelForSeq2SeqLM")
@patch("rag_core.pipeline")
def test_generator_falls_back_to_local_when_no_key(
    mock_pipe, mock_model, mock_tok, mock_genai, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # pipeline returns list with dict containing generated_text
    mock_pipe.return_value = lambda prompt, max_new_tokens, do_sample: [
        {"generated_text": " local out "}
    ]

    g = rc.Generator(use_gemini=True, local_model="local-model")
    out = g.generate("p")

    assert out == "local out"
    mock_tok.from_pretrained.assert_called_with("local-model")
    mock_model.from_pretrained.assert_called_with("local-model")


# ---------- Retriever tests ----------
@patch("rag_core.wikipedia")
@patch("rag_core.BM25Okapi")
def test_retriever_basic_flow(mock_bm25, mock_wiki):
    r = rc.Retriever(k_titles=2, k_sents=3, sleep=0)
    mock_wiki.search.return_value = ["T1", "T2"]
    summaries = {
        "T1": "Alpha is first. Beta is second. Gamma is third.",
        "T2": "Delta appears. Epsilon follows. Zeta ends.",
    }

    def summary_side_effect(title):
        return summaries[title]

    mock_wiki.summary.side_effect = summary_side_effect

    # Fake BM25: just return ascending scores by sentence index
    def fake_get_scores(tokens):
        # there are 6 sentences total in summaries
        return [i for i in range(6)]

    mock_bm25.return_value = MagicMock(get_scores=fake_get_scores)

    out = r.retrieve("alpha", max_chars=1000)

    assert "context" in out and "sources" in out
    assert len(out["sources"]) == 2
    # k_sents=3 -> should select 3 sentences
    assert (
        len(
            rc.Retriever._split_sentences(r, summaries["T1"])
            + rc.Retriever._split_sentences(r, summaries["T2"])
        )
        >= 6
    )
    assert len(out["context"].split(".")) >= 3


@patch("rag_core.wikipedia")
def test_retriever_handles_no_results(mock_wiki):
    r = rc.Retriever(k_titles=2, k_sents=2, sleep=0)
    mock_wiki.search.return_value = []
    out = r.retrieve("no results")
    assert out == {"context": "", "sources": []}


# ---------- Verifier tests ----------
@patch("rag_core.util")
@patch("rag_core.SentenceTransformer")
def test_verifier_similarity_and_verdict(
    mock_st_cls, mock_util, fake_sentence_transformer
):
    mock_st_cls.return_value = fake_sentence_transformer
    mock_util.cos_sim.side_effect = lambda a, b: [
        [types.SimpleNamespace(item=lambda: 0.8)]
    ]

    v = rc.Verifier(threshold=0.7)
    score = v.similarity_score("abc", "abcdef")
    label, s = v.verdict("abc", "abcdef")

    assert 0.0 < score <= 1.0
    assert label == "Factual"
    assert pytest.approx(s, 1e-6) == score


@patch("rag_core.util")
@patch("rag_core.SentenceTransformer")
def test_verifier_empty_inputs(mock_st_cls, mock_util, fake_sentence_transformer):
    mock_st_cls.return_value = fake_sentence_transformer
    v = rc.Verifier(threshold=0.9)

    assert v.similarity_score("", "x") == 0.0
    assert v.similarity_score("x", "") == 0.0
    label, s = v.verdict("", "")
    assert label == "Hallucinated" and s == 0.0
