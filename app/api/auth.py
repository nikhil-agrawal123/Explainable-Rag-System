import secrets
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

from fastapi import Header, HTTPException, status

from app.core.config import settings


@dataclass(frozen=True)
class AuthenticatedUser:
    email: str


_active_tokens: Dict[str, AuthenticatedUser] = {}
_token_lock = Lock()


def _expected_credentials() -> tuple[Optional[str], Optional[str]]:
    return settings.AUTH_EMAIL, settings.AUTH_PASSWORD


def authenticate(email: str, password: str) -> str:
    expected_email, expected_password = _expected_credentials()

    if not expected_email or not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    if email != expected_email or password != expected_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )

    token = secrets.token_urlsafe(32)
    with _token_lock:
        _active_tokens[token] = AuthenticatedUser(email=email)
    return token


def logout(token: str) -> None:
    with _token_lock:
        _active_tokens.pop(token, None)


def get_current_user(authorization: str = Header(default="")) -> AuthenticatedUser:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization token.",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization token.",
        )

    with _token_lock:
        user = _active_tokens.get(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You must be logged in to access this endpoint.",
        )

    return user
