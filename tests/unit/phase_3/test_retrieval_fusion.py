from src.retrieval.fusion import rrf_fuse
from types import SimpleNamespace


def make_item(cid):
    return SimpleNamespace(chunk_id=cid, payload={"text": "chunk %s" % cid})


def test_rrf_basic():
    a = [make_item("a"), make_item("b"), make_item("c")]
    b = [make_item("b"), make_item("d"), make_item("a")]
    fused = rrf_fuse([a, b], k_smooth=60, weights=[1.0, 1.0])
    assert len(fused) >= 1
    ids = [f.chunk_id for f in fused]
    assert "a" in ids and "b" in ids
