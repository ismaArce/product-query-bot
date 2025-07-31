import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _post_query(user_id: str, query: str):
    return client.post(
        "/query",
        json={"user_id": user_id, "query": query},
        headers={"Content-Type": "application/json"},
    )


@pytest.mark.integration
def test_validation_error_on_missing_fields():
    """Test that missing user_id or query returns validation error."""
    resp = client.post("/query", json={})
    assert resp.status_code in (400, 422), resp.text


@pytest.mark.integration
def test_elliptic_question_triggers_clarification():
    """Test that elliptical question without context triggers product clarification."""
    uid = "t-clarify-1"
    resp = _post_query(uid, "and what is the price?")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    answer = (data.get("answer") or "").lower()
    assert any(
        key in answer
        for key in [
            "which product",
            "what product",
        ]
    ), f"Expected a clarification about product, got: {answer!r}"


@pytest.mark.integration
def test_product_then_price_does_not_reask():
    """Test that asking price after specifying product doesn't re-ask for product."""
    uid = "t-product-1"
    resp1 = _post_query(uid, "Annibale Colombo Sofa")
    assert resp1.status_code == 200, resp1.text

    resp2 = _post_query(uid, "and what is the price?")
    assert resp2.status_code == 200, resp2.text
    ans2 = (resp2.json().get("answer") or "").lower()

    assert all(
        phrase not in ans2
        for phrase in [
            "which product",
            "what product",
        ]
    ), f"Should not re-ask for product; got: {ans2!r}"


@pytest.mark.integration
def test_thread_isolation_new_user_needs_clarification_again():
    """Test conversation isolation by user_id - new user needs clarification again."""
    uid_a = "t-isolation-a"
    _ = _post_query(uid_a, "Some Fancy Chair Model X")
    resp_a2 = _post_query(uid_a, "and what is the price?")
    assert resp_a2.status_code == 200, resp_a2.text
    ans_a2 = (resp_a2.json().get("answer") or "").lower()
    assert all(
        phrase not in ans_a2
        for phrase in [
            "which product",
            "what product",
        ]
    ), f"Thread A: should not re-ask for product; got: {ans_a2!r}"

    uid_b = "t-isolation-b"
    resp_b1 = _post_query(uid_b, "and what is the price?")
    assert resp_b1.status_code == 200, resp_b1.text
    ans_b1 = (resp_b1.json().get("answer") or "").lower()
    assert any(
        key in ans_b1
        for key in [
            "which product",
            "what product",
        ]
    ), f"Thread B: should ask for clarification; got: {ans_b1!r}"
