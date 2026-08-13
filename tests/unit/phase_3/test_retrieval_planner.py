from src.retrieval.planner import QueryPlanner


def test_planner_basic():
    qp = QueryPlanner()
    r = qp.plan("What parameter is required for POST /v1/orders?")
    assert r["intent"] in ("factual", "ambiguous")
    assert "/v1/orders" in (r.get("entity") or "")

    r2 = qp.plan("Compare v1 and v2 of the API for currency type")
    assert r2["intent"] == "comparative"

    r3 = qp.plan("Where is the security guide?")
    assert r3["intent"] == "navigational"
