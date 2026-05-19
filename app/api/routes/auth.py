from fastapi import APIRouter, Depends, Header, HTTPException, status
from app.api.auth import create_access_token, get_current_user
from supabase import Client, create_client
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter(tags=["Auth"])

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
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if getattr(response, "user", None) is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {e}",
        )
    user_id = response.user.id if getattr(response, "user", None) else ""
    token = create_access_token(email=email, user_id=user_id)
    return {"status": "success", "access_token": token, "token_type": "bearer"}


@router.post("/logout/", summary="Log out (client-side token discard)")
def logout_endpoint(authorization: str = Header(default="")):
    try:
        if authorization.startswith("Bearer "):
            supabase.auth.sign_out()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Logout failed: {e}",
        )
    return {"status": "success", "message": "Token discarded. Remove it from client storage."}


@router.get("/me/", summary="Return the authenticated user")
def me(current_user=Depends(get_current_user)):
    return {"status": "success", "email": current_user.email}
