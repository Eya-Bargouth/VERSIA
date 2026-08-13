from src.retrieval.reranker import Reranker


def test_reranker_noop():
    r = Reranker()
    candidates = [
        {'chunk_id':'a','payload':{'text':'short'}},
        {'chunk_id':'b','payload':{'text':'a much longer text example'}}
    ]
    out = r.rerank('query', candidates, top_k=2)
    # fallback sorts by length ascending -> short first
    assert out[0]['chunk_id'] == 'a'
