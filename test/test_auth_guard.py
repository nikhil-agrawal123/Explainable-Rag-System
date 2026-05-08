from app.core.config import settings


def test_api_requires_login_before_access(test_client, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret123", raising=False)

    unauthorized = test_client.post(
        "/api/v3/query/",
        params={"query": "How do stars form?"},
    )
    assert unauthorized.status_code == 401
