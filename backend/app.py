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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

@app.get("/frontend/{filename}")
async def serve_frontend_files(filename: str):
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"detail": "Not Found"}

# Global variables for model and vectorizer
vectorizer = None
model = None
label_encoder = None
df = None

def load_and_prepare_data():
    """Load dataset and prepare for training"""
    global df
    
    data_path = os.path.join("backend", "train.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print("⚠️ Skipped invalid JSON line")
    
    df = pd.DataFrame(data)
    
    # Clean data
    df = df.dropna(subset=['input_text', 'output_text'])
    df['input_text'] = df['input_text'].str.lower().str.strip()
    df['output_text'] = df['output_text'].str.strip()
    
    print(f"✅ Loaded {len(df)} symptom-diagnosis pairs from dataset.")
    return df

def train_model():
    """Train the ML model on the dataset"""
    global vectorizer, model, label_encoder, df
    
    print("\n🔄 Training machine learning model...")
    
    # Prepare data
    X = df['input_text'].values
    y = df['output_text'].values
    
    # Create label mapping (for diagnosis names)
    unique_diagnoses = list(set(y))
    label_encoder = {diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)}
    reverse_label_encoder = {idx: diagnosis for diagnosis, idx in label_encoder.items()}
    
    # Encode labels
    y_encoded = [label_encoder[diagnosis] for diagnosis in y]
    
    print(f"📊 Dataset: {len(X)} samples, {len(unique_diagnoses)} unique diagnoses")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Naive Bayes classifier
    print("🎯 Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Accuracy: {accuracy * 100:.2f}%")
    print(f"🧠 Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    # Save model and vectorizer
    model_dir = os.path.join("backend", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump((label_encoder, reverse_label_encoder), f)
    
    print("💾 Model saved to backend/models/")
    
    return reverse_label_encoder

def load_trained_model():
    """Load pre-trained model if exists"""
    global vectorizer, model, label_encoder
    
    model_dir = os.path.join("backend", "models")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(encoder_path, "rb") as f:
            label_encoder, reverse_label_encoder = pickle.load(f)
        
        print("✅ Loaded pre-trained model from disk")
        return reverse_label_encoder
    
    return None

def predict_diagnosis(user_input, top_n=3):
    """Predict diagnosis using trained ML model"""
    global vectorizer, model
    
    # Transform user input
    user_tfidf = vectorizer.transform([user_input.lower().strip()])
    
    # Get probability predictions for all classes
    probabilities = model.predict_proba(user_tfidf)[0]
    
    # Get top N predictions
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    
    predictions = []
    for idx in top_indices:
        confidence = probabilities[idx] * 100
        if confidence > 5:  # Only show predictions with >5% confidence
            predictions.append({
                'diagnosis': reverse_label_encoder[idx],
                'confidence': float(confidence),
                'model_type': 'Naive Bayes Classifier'
            })
    
    return predictions

def generate_ml_response(user_input, predictions):
    """Generate a clean, concise formatted ML output"""
    if not predictions:
        return {
            "response": (
                f"Based on your symptoms: \"{user_input}\", I couldn't confidently match them "
                "to a known condition in my database.\n\n"
                "If your symptoms persist or worsen, please consult a doctor."
            ),
            "predictions": []
        }

    # Start response
    response_text = f"<b>Based on your symptoms:</b> \"{user_input}\"<br><br>"
    response_text += "<b>🩺 Possible Conditions:</b><br>"

    for pred in predictions:
        # Determine severity level
        confidence = pred["confidence"]
        if confidence > 60:
            severity = "High severity"
        elif confidence > 30:
            severity = "Moderate severity"
        else:
            severity = "Low severity"

        # Add each disease line
        response_text += f"  <li><b>{pred['diagnosis']}</b> — Confidence: {confidence:.1f}% ({severity})</li><br>"

    response_text += (
        "\n<i><b>Recommendation:</i></b><br>"
        """If symptoms <b>persist or worsen</b>, please <a href= "/frontend/doctors.html" target="_blank"> <b> consult a doctor</b></a> for professional medical advice."""
    )

    return {
        "response": response_text,
        "predictions": predictions,
        "ml_model": "Naive Bayes",
        "training_samples": len(df)
    }


# Input model
class SymptomRequest(BaseModel):
    symptoms: str

@app.on_event("startup")
async def startup_event():
    """Initialize and train model on startup"""
    global reverse_label_encoder
    
    # Load data
    load_and_prepare_data()
    
    # Try to load pre-trained model
    reverse_label_encoder = load_trained_model()
    
    # If no pre-trained model, train new one
    if reverse_label_encoder is None:
        reverse_label_encoder = train_model()

@app.post("/analyze")
async def analyze_symptoms(request: SymptomRequest):
    user_input = request.symptoms.strip()
    
    if not user_input:
        raise HTTPException(status_code=400, detail="Symptoms cannot be empty.")
    
    if len(user_input) < 5:
        raise HTTPException(status_code=400, detail="Please provide more detailed symptoms.")
    
    try:
        # Get ML predictions
        predictions = predict_diagnosis(user_input, top_n=3)
        
        # Generate response
        result = generate_ml_response(user_input, predictions)
        
        return result
    
    except Exception as e:
        print(f"❌ Error analyzing symptoms: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing symptoms: {str(e)}"
        )

@app.post("/retrain")
async def retrain_model():
    """Endpoint to retrain the model"""
    try:
        global reverse_label_encoder
        reverse_label_encoder = train_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "training_samples": len(df)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retraining model: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dataset_size": len(df) if df is not None else 0,
        "model_trained": model is not None,
        "model_type": "Naive Bayes Classifier",
        "message": "ML Medical Symptom Analyzer is running"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the trained model"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    return {
        "model_type": "Multinomial Naive Bayes",
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
        "training_samples": len(df),
        "unique_diagnoses": len(label_encoder),
        "features": "TF-IDF (unigrams + bigrams)",
        "status": "trained"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)