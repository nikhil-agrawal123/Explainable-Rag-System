import secrets
from dataclasses import dataclass
from threading import Lock
from typing import Dict

from fastapi import Header, HTTPException, status



@dataclass(frozen=True)
class AuthenticatedUser:
    email: str


_active_tokens: Dict[str, AuthenticatedUser] = {}
_token_lock = Lock()


def authenticate(email: str) -> str:
    
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
