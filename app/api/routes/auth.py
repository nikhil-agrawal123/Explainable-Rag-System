from fastapi import APIRouter, Depends, Form, Header, HTTPException, status
from app.api.auth import authenticate, logout, get_current_user
from supabase import Client, create_client
from dotenv import load_dotenv
from pydantic import SecretStr
from bcrypt import hashpw, gensalt
import os

load_dotenv()
router = APIRouter()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


@router.post("/signup", summary="Sign up for an account (not implemented, placeholder)")
def signup(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
):
    if not first_name or not last_name or not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All fields (first_name, last_name, email, password) are required.",
        )

    try:
        response = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "first_name": first_name,
                        "last_name": last_name,
                        "email": email,
                    }
                },
            }
        )

        if getattr(response, "user", None) is None:
            return {"status": "error", "message": "Signup failed."}

        return {"status": "success", "message": "User created successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Error creating user: {e}"}



@router.post("/login/", summary="Log in and receive an access token")
def login(
    email: str,
    password: str
):
    
    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required.",
        )
        
    try:
        respoonse = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if getattr(respoonse, "user", None) is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {e}",
        )
    token = authenticate(email, password)
    return {"status": "success", "access_token": token, "token_type": "bearer"}


@router.post("/logout/", summary="Log out and revoke the access token")
def logout_endpoint(authorization: str = Header(default="")):
    if authorization.startswith("Bearer "):
        logout(authorization.removeprefix("Bearer ").strip())
    return {"status": "success"}


@router.get("/me/", summary="Return the authenticated user")
def me(current_user=Depends(get_current_user)):
    return {"status": "success", "email": current_user.email}
