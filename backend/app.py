from fastapi import FastAPI, Request
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import os
import random

app = FastAPI()

# Static and frontend directories setup
front_end = os.path.join(os.path.dirname(__file__), '../frontend')
static_dir = os.path.join(front_end, 'static')

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print("⚠️ Front-end directory not found:", front_end)

@app.get("/")
async def root():
    return FileResponse(os.path.join(front_end, 'index.html'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset safely from JSON Lines file
data = []
with open("backend/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print("⚠️ Skipped bad line:", line[:80])

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} rows from dataset.")

# Load OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client with your API key
client = OpenAI(api_key=api_key)

@app.post("/analyze")
async def analyze(request: Request):
    req_data = await request.json()
    user_input = req_data.get("message", "").lower()
    user_words = set(user_input.split())

    # Match relevant rows from dataset by symptom words intersection
    possible_matches = df[df['input_text'].str.contains(
        "|".join(user_input.split()), case=False, na=False
    )]


    if not possible_matches.empty:
        diagnosis = random.choice(possible_matches['output_text'].tolist())
        base_response = f"Based on your symptoms, a possible diagnosis could be: {diagnosis}."
    else:
        base_response = "I'm sorry, I couldn't find a matching diagnosis in the dataset."
        
    # Sample top candidates to provide context for GPT
    candidates = possible_matches.sample(min(5, len(possible_matches)))
    summarized_context = "\n".join(
        [f"Symptoms: {row['input_text']}\nDiagnosis: {row['output_text']}" for _, row in candidates.iterrows()]
    )

    # Compose prompt for GPT
    prompt = f"""
You are a medical assistant trained to identify possible conditions from symptoms.
The user described: "{user_input}"

Here are examples of symptom-diagnosis pairs from real data:
{summarized_context}

Based on this and your medical reasoning, list up to 3 possible conditions (common illnesses only),
then explain briefly why those might fit.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.choices[0].message.content.strip()
        return {"response": ai_text}
    except Exception as e:
        print("⚠️ OpenAI error:", e)
        return {"response": "Sorry, I couldn't analyze your symptoms right now."}
