import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sentence_transformers")

import numpy as np

from voussoir.llm.sentence_transformers_embedder import SentenceTransformersEmbedder


def test_provider_name_and_default_model():
    e = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
    assert e.name == "sentence-transformers"
    assert e.default_model == "all-MiniLM-L6-v2"


def test_model_load_is_lazy():
    """Construction must NOT load the model — that's expensive and
    happens on first embed call only."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        SentenceTransformersEmbedder()
        mock_st.assert_not_called()


async def test_embed_single_round_trips_through_model():
    fake_model = MagicMock()
    fake_model.encode = MagicMock(return_value=np.array([0.1] * 384))
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        e = SentenceTransformersEmbedder()
        vec = await e.embed_single("hello")
    assert len(vec) == 384
    fake_model.encode.assert_called_once_with("hello")


async def test_embed_batch_returns_one_vector_per_text():
    fake_model = MagicMock()
    fake_model.encode = MagicMock(return_value=np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384]))
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        e = SentenceTransformersEmbedder()
        out = await e.embed(["a", "b", "c"])
    assert len(out.embeddings) == 3
    assert all(len(v) == 384 for v in out.embeddings)


async def test_model_loaded_once_across_calls():
    """The model is cached after first load — subsequent calls reuse it."""
    fake_model = MagicMock()
    fake_model.encode = MagicMock(return_value=np.array([0.0] * 384))
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as mock_st:
        e = SentenceTransformersEmbedder()
        await e.embed_single("x")
        await e.embed_single("y")
        await e.embed_single("z")
    assert mock_st.call_count == 1


@pytest.mark.skipif(
    "VOUSSOIR_RUN_EMBEDDING_MODEL_TESTS" not in os.environ,
    reason="opt-in: downloads ~80MB model weights on first run",
)
async def test_real_model_produces_384_dim_vector():
    e = SentenceTransformersEmbedder()
    vec = await e.embed_single("the quick brown fox")
    assert len(vec) == 384
    assert isinstance(vec[0], float)
