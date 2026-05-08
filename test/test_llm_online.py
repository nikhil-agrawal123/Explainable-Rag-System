import socket
from urllib.parse import urlparse
import pytest

from app.core.config import settings


def test_ollama_host_reachable():
    """Verify the configured OLLAMA_HOST is reachable via TCP.

    If `OLLAMA_HOST` is not configured the test is skipped.
    """
    host = settings.OLLAMA_HOST
    if not host:
        pytest.skip("OLLAMA_HOST not configured")

    parsed = urlparse(host)
    hostname = parsed.hostname or parsed.path
    port = parsed.port or 11434

    try:
        with socket.create_connection((hostname, port), timeout=3):
            assert True
    except Exception as e:
        pytest.fail(f"Cannot connect to OLLAMA host {hostname}:{port} — {e}")
