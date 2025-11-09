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
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

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
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Global variables
vectorizer = None
model = None
label_encoder = None
reverse_label_encoder = None
df = None
home_remedies = {}

# Load Home Remedies JSONL
def load_home_remedies():
    global home_remedies
    remedies_file = "home_remedies.jsonl"

    if not os.path.exists(remedies_file):
        print("⚠️ No home_remedies.jsonl found.")
        return

    with open(remedies_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                symptom = data["symptom"].lower()
                remedy = data["remedy"]

                if symptom not in home_remedies:
                    home_remedies[symptom] = []

                home_remedies[symptom].append(remedy)

            except:
                continue

    print(f"✅ Loaded {len(home_remedies)} home remedies.")

# Preprocess input: fix typos & synonyms
def preprocess_input(text):
    text = text.lower().strip()
    corrections = {
        "feve": "fever",
        "joint ache": "joint pain",
        "joint hurting": "joint pain",
        "urinate pain": "burning while urinating",
        "stomach ache": "abdominal pain",
        "head ache": "headache",
        # Add more as needed
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# Load and prepare dataset
def load_and_prepare_data():
    global df
    data_path = "train.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    df = pd.DataFrame(data)
    df = df.dropna(subset=['input_text', 'output_text'])
    df['input_text'] = df['input_text'].str.lower().str.strip()
    df['output_text'] = df['output_text'].str.strip()
    return df

# Train model with calibration
def train_model():
    global vectorizer, model, label_encoder, reverse_label_encoder, df

    X = df['input_text'].values
    y = df['output_text'].values

    unique_diagnoses = list(set(y))
    label_encoder = {diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)}
    reverse_label_encoder = {idx: diagnosis for diagnosis, idx in label_encoder.items()}
    y_encoded = [label_encoder[d] for d in y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    base_model = MultinomialNB(alpha=0.1)
    base_model.fit(X_train_tfidf, y_train)

    # Calibrate the model for better confidence percentages
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_test_tfidf, y_test)
    model = calibrated_model

    accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
    print(f"✅ Model trained. Accuracy on validation set: {accuracy*100:.2f}%")

    # Save vectorizer and model
    model_dir = os.path.join("backend", "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump((label_encoder, reverse_label_encoder), f)

    return reverse_label_encoder

# Load trained model
def load_trained_model():
    global vectorizer, model, label_encoder, reverse_label_encoder
    model_dir = os.path.join("backend", "models")
    try:
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder, reverse_label_encoder = pickle.load(f)
        return reverse_label_encoder
    except:
        return None

# Predict diagnosis
def predict_diagnosis(user_input, top_n=3):
    user_input = preprocess_input(user_input)  # ✅ Preprocess input before prediction
    user_tfidf = vectorizer.transform([user_input])
    probabilities = model.predict_proba(user_tfidf)[0]

    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    predictions = []

    for idx in top_indices:
        pred_conf = probabilities[idx] * 100
        if pred_conf > 5:
            predictions.append({
                'diagnosis': reverse_label_encoder[idx],
                'confidence': float(pred_conf)
            })

    return predictions

# Generate response with remedies
def generate_ml_response(user_input, predictions):
    if not predictions:
        return {"response": f"Based on your symptoms: \"{user_input}\" I could not match any condition."}

    lines = []
    lines.append(f'Based on your symptoms: "{user_input}"\n')
    lines.append("🩺 Possible Conditions:")

    for p in predictions:
        cond = p['diagnosis']
        conf = p['confidence']
        severity = (
            "High severity" if conf > 60 else
            "Moderate severity" if conf > 30 else
            "Low severity"
        )
        lines.append(f"• {cond} — Confidence: {conf:.1f}% ({severity})")

    # Match remedies
    matched_remedies = []
    for symptom, remedy_list in home_remedies.items():
        if symptom in user_input.lower():
            matched_remedies.extend(remedy_list)

    if not matched_remedies:
        matched_remedies = [
            "Rest well",
            "Drink plenty of fluids",
            "Monitor symptoms"
        ]

    lines.append("\n💡 Home Remedies:")
    for r in matched_remedies[:3]:
        lines.append(f"- {r}")

    lines.append("\n🔔 Recommendation:")
    lines.append("If symptoms persist or worsen, please consult a doctor.")

    return {"response": "\n".join(lines)}

# Pydantic model
class SymptomRequest(BaseModel):
    symptoms: str

# Startup event
@app.on_event("startup")
async def startup_event():
    global reverse_label_encoder
    load_home_remedies()
    load_and_prepare_data()
    reverse_label_encoder = load_trained_model()
    if reverse_label_encoder is None:
        reverse_label_encoder = train_model()

# Analyze endpoint
@app.post("/analyze")
async def analyze_symptoms(request: SymptomRequest):
    try:
        predictions = predict_diagnosis(request.symptoms, top_n=3)
        return generate_ml_response(request.symptoms, predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
