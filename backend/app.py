from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
from backend.db import get_db
from backend.auth import router as auth_router 

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory
BASE_DIR = os.path.dirname(__file__)
FRONTEND_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../frontend"))           # frontend/
SIGNUP_DIR = os.path.join(FRONTEND_ROOT, "sign-up")                               # frontend/sign-up/
STATIC_DIR = os.path.join(SIGNUP_DIR, "static-up")                                # frontend/sign-up/static-up/

# Mount static folder
app.mount("/static-up", StaticFiles(directory=STATIC_DIR), name="static-up")
# Add this above your route definitions
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_ROOT, "static")), name="static")
app.mount("/static-reg",StaticFiles(directory=os.path.join(FRONTEND_ROOT,"register","static-reg")),name="static-reg")
app.mount("/register", StaticFiles(directory=os.path.join(FRONTEND_ROOT, "register")), name="register")
app.mount("/sign-up", StaticFiles(directory=os.path.join(FRONTEND_ROOT,"sign-up")),name="sign-up")
app.mount("/frontend", StaticFiles(directory=FRONTEND_ROOT), name="frontend")


app.include_router(auth_router)

# ---------------------- ROUTES ----------------------

# Default route → sign-up page
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(SIGNUP_DIR, "sign-up.html"))

# Login button → main index.html (outside sign-up folder)
@app.get("/index.html")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_ROOT, "index.html"))

# Serve other HTML pages directly
@app.get("/sign-up")
def sign_up():
    return FileResponse("frontend/sign-up/sign-up.html")

REGISTER_DIR = os.path.join(FRONTEND_ROOT, "register")

@app.get("/registerl")
async def serve_register():
    return FileResponse(os.path.join(REGISTER_DIR, "register.html"))


# Serve additional signup-related files if needed
@app.get("/frontend/{filename}")
async def serve_frontend_files(filename: str):
    file_path = os.path.join(SIGNUP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"detail": "Not Found"}

# Global variables
vectorizer = None
model = None
label_encoder = None
df = None

# ✅ HOME REMEDIES LOADING (JSONL Correct Format)
home_remedies = []

def load_home_remedies():
    global home_remedies
    path = os.path.join(os.path.dirname(__file__), "home_remedies.jsonl")
    home_remedies = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                home_remedies.append(json.loads(line.strip()))
            except:
                continue

def get_3_home_remedies(symptom_text):
    symptom_text = symptom_text.lower()

    matched = [item["remedy"] for item in home_remedies if item["symptom"].lower() in symptom_text]

    if len(matched) >= 3:
        return matched[:3]

    extras = [item["remedy"] for item in home_remedies if item["remedy"] not in matched]
    random.shuffle(extras)

    while len(matched) < 3 and extras:
        matched.append(extras.pop())

    return matched[:3]

# ✅ MODEL LOADING + TRAINING
def load_and_prepare_data():
    global df
    
    data_path = os.path.join(os.path.dirname(__file__), "train.jsonl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['input_text', 'output_text'])
    df['input_text'] = df['input_text'].str.lower().str.strip()
    df['output_text'] = df['output_text'].str.strip()

    return df

def train_model():
    global vectorizer, model, label_encoder, df
    
    X = df['input_text'].values
    y = df['output_text'].values
    
    unique_diagnoses = list(set(y))
    label_encoder = {d: i for i, d in enumerate(unique_diagnoses)}
    reverse_label_encoder = {v: k for k, v in label_encoder.items()}
    
    y_encoded = [label_encoder[d] for d in y]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    global reverse_label_encoder_global
    reverse_label_encoder_global = reverse_label_encoder

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    return reverse_label_encoder

def load_trained_model():
    global vectorizer, model, label_encoder
    
    model_dir = "backend/models"
    if os.path.exists(f"{model_dir}/vectorizer.pkl"):
        vectorizer = pickle.load(open(f"{model_dir}/vectorizer.pkl", "rb"))
        model = pickle.load(open(f"{model_dir}/model.pkl", "rb"))
        label_encoder, reverse_label_encoder = pickle.load(open(f"{model_dir}/label_encoder.pkl", "rb"))
        return reverse_label_encoder
    return None

def predict_diagnosis(user_input, top_n=3):
    user_tfidf = vectorizer.transform([user_input.lower().strip()])
    probs = model.predict_proba(user_tfidf)[0]
    top_idx = np.argsort(probs)[-top_n:][::-1]

    predictions = []
    for idx in top_idx:
        predictions.append({
            'diagnosis': reverse_label_encoder_global[idx],
            'confidence': float(probs[idx] * 100)
        })
    return predictions

def generate_ml_response(user_input, predictions):
    
    remedies = get_3_home_remedies(user_input)

    response_text = f"<b>Based on your symptoms:</b> \"{user_input}\"<br>"
    response_text += f"<b>🩺 Possible Conditions:</b><ul>"
    for pred in predictions:
        c = pred["confidence"]
        severity = "High" if c > 60 else "Moderate" if c > 30 else "Low"
        response_text += f"<li>{pred['diagnosis']} — Confidence: {c:.1f}% ({severity} severity)</li><br>"
    response_text += "</ul>"

    response_text += "<b>💡 Home Remedies:</b><ul>"
    for r in remedies:
        response_text += f"<li>{r}</li><br>"
    response_text += "</ul>"



    response_text += "🔔<b><i> Recommendation:</i></b><br>If symptoms persist or worsen, please consult a doctor."

    return {"response": response_text}


class SymptomRequest(BaseModel):
    symptoms: str

@app.on_event("startup")
async def startup_event():
    global reverse_label_encoder_global
    load_home_remedies()
    load_and_prepare_data()
    reverse_label_encoder_global = load_trained_model()
    if reverse_label_encoder_global is None:
        reverse_label_encoder_global = train_model()

@app.post("/analyze")
async def analyze_symptoms(request: SymptomRequest):
    predictions = predict_diagnosis(request.symptoms, top_n=3)
    return generate_ml_response(request.symptoms, predictions)

@app.get("/health")
async def health_check():
    return {"status": "running", "dataset_size": len(df)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)