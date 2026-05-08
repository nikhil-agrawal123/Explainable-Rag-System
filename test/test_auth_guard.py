from app.core.config import settings


def test_api_requires_login_before_access(test_client, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_EMAIL", "user@example.com", raising=False)
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret123", raising=False)

    unauthorized = test_client.post(
        "/api/v3/query/",
        params={"query": "How do stars form?"},
    )
    assert unauthorized.status_code == 401
    assert "logged in" in unauthorized.json()["detail"].lower() or "authorization" in unauthorized.json()["detail"].lower()

    login = test_client.post(
        "/api/v3/login/",
        data={"email": "user@example.com", "password": "secret123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    authorized = test_client.post(
        "/api/v3/query/",
        params={"query": "How do stars form?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["status"] == "success"
