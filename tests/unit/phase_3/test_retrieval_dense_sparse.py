import pytest
from src.retrieval.dense_search import DenseSearch
from src.retrieval.sparse_search import SparseSearch
from types import SimpleNamespace


class DummyStore:
    def __init__(self):
        pass
    def search_dense(self, v, filter=None, k=10):
        return [SimpleNamespace(chunk_id='c1', payload={'text':'hello'}, score=0.9)]
    def search_sparse(self, s, filter=None, k=10):
        return [SimpleNamespace(chunk_id='s1', payload={'text':'sparse'}, score=0.8)]


def test_dense_search():
    store = DummyStore()
    ds = DenseSearch(store)
    res = ds.search_dense([0.1]*1024)
    assert len(res) == 1
    assert res[0].chunk_id == 'c1'


def test_sparse_search():
    store = DummyStore()
    ss = SparseSearch(store)
    res = ss.search_sparse({1:0.5})
    assert len(res) == 1
    assert res[0].chunk_id == 's1'
