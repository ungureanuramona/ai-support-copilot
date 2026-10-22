
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)

VALID_TICKET = {
    "title": "Users cannot log in",
    "description": (
        "Multiple users cannot access the application "
        "after scheduled maintenance."
    ),
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ai-support-copilot",
    }


def test_analyze_ticket_success():
    documents = [
        {
            "source": "authentication_incidents.txt",
            "content": "Check authentication service logs.",
        }
    ]

    relevant_documents = [
        {
            "source": "authentication_incidents.txt",
            "content": "Check authentication service logs.",
            "score": 0.85,
        }
    ]

    with (
        patch("api.load_documents", return_value=documents),
        patch("api.get_embedding_model", return_value=object()),
        patch(
            "api.find_relevant_documents",
            return_value=relevant_documents,
        ),
        patch(
            "api.generate_support_plan",
            return_value="## Recommended next steps\nCheck logs.",
        ),
    ):
        response = client.post("/analyze", json=VALID_TICKET)

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "Knowledge base match"
    assert "Check logs" in data["support_plan"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == (
        "authentication_incidents.txt"
    )
    assert data["sources"][0]["similarity"] == 0.85


def test_analyze_ticket_validation():
    response = client.post(
        "/analyze",
        json={
            "title": "Hi",
            "description": "Too short",
        },
    )

    assert response.status_code == 422


def test_analyze_ticket_ai_unavailable():
    with (
        patch("api.load_documents", return_value=[]),
        patch(
            "api.get_embedding_model",
            side_effect=ConnectionError("Model unavailable"),
        ),
    ):
        response = client.post("/analyze", json=VALID_TICKET)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The analysis service is temporarily unavailable."
    )


def test_analyze_ticket_no_relevant_documents():
    with (
        patch("api.load_documents", return_value=[]),
        patch("api.get_embedding_model", return_value=object()),
        patch("api.find_relevant_documents", return_value=[]),
        patch(
            "api.generate_support_plan",
            return_value=(
                "## Category\n"
                "Unknown - Insufficient knowledge base coverage\n"
                "## Sources\n"
                "No sufficiently relevant documents found."
            ),
        ),
    ):
        response = client.post("/analyze", json=VALID_TICKET)

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == (
        "Insufficient knowledge base coverage"
    )
    assert data["sources"] == []
    assert "Insufficient knowledge base coverage" in (
        data["support_plan"]
    )
