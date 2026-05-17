def test_api_requires_login_before_access(test_client):
    unauthorized = test_client.post(
        "/api/v3/query/",
        params={"query": "How do stars form?"},
    )
    assert unauthorized.status_code == 401
