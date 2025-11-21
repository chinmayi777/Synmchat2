from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import bcrypt
import os
from fastapi import Form
from backend.db import get_db
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")


class RegisterModel(BaseModel):
    first_name: str
    last_name:str
    email: str
    password: str

@router.post("/register")
def register_user(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirmPassword: str = Form(...) # confirmPassword should be handled by the frontend, but is fine here
):
    # Check 1: Client-side password mismatch defense (optional, but good)
    if password != confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    db = get_db()
    cursor = db.cursor(dictionary=True)

    # Check 2: Check for existing user by email
    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (email,)
    )
    existing_user = cursor.fetchone()

    if existing_user:
        cursor.close()
        db.close()
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Check 3: FIX SQL INSERT QUERY
    cursor.execute(
        # Removed the extra %s placeholder
        "INSERT INTO users (first_name, last_name, email, password) VALUES (%s,%s,%s,%s)",
        (first_name, last_name, email, hashed_pw)
    )
    db.commit()
    cursor.close()
    db.close()

    return RedirectResponse(url="/index.html", status_code=303)


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email=%s", (form_data.username,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        db.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not bcrypt.checkpw(form_data.password.encode('utf-8'),
                          user["password"].encode("utf-8")):
        cursor.close()
        db.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token_data = {
        "user_id": user["id"],
        "email": user["email"],
        "exp": datetime.utcnow() + timedelta(hours=12)
    }

    token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)

    cursor.close()
    db.close()

    return RedirectResponse(url="/index.html", status_code=303)
