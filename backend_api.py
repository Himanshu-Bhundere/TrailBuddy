"""
AI Travel Itinerary Generator - FastAPI Backend
Web API for generating travel itineraries using OpenAI
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(title="Travel Itinerary Generator")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ItineraryRequest(BaseModel):
    place: str
    duration: int
    theme: List[str]
    number_of_people: int
    budget_level: str
    interests: List[str]
    pace: str
    accommodation_area: str
    transport_preference: str
    food_preference: str
    constraints: List[str] = []
    season_dates: List[str] = []

# --------------------------------------------------
# SAFE JSON EXTRACTION
# --------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in OpenAI response")
    return json.loads(text[start:end + 1])

# --------------------------------------------------
# GENERATE ITINERARY
# --------------------------------------------------

def generate_itinerary(request: ItineraryRequest) -> dict:
    """
    Generate travel itinerary using OpenAI
    """
    
    PROMPT = f"""
Act as an expert travel consultant with 10+ years of experience in luxury and local-authentic travel planning. Create a highly detailed, optimized travel itinerary based on the following constraints:

DESTINATION: {request.place}
TRIP DURATION: {request.duration} days
THEME: {", ".join(request.theme)}
TRAVELERS: {request.number_of_people}
BUDGET LEVEL: {request.budget_level}
INTERESTS: {", ".join(request.interests)}
PACE: {request.pace}
ACCOMMODATION AREA: {request.accommodation_area}
TRANSPORT: {request.transport_preference}
FOOD: {request.food_preference}
CONSTRAINTS: {", ".join(request.constraints)}
SEASON: {", ".join(request.season_dates)}

RETURN ONLY VALID JSON.
DO NOT include markdown, explanations, tables, or extra text.

THE JSON MUST MATCH THIS SCHEMA EXACTLY:

{{
  "destination": "City, Country",
  "duration": {request.duration},
  "budget_level": "string",
  "activities": [],
  "days": [
    {{
      "day": 1,
      "places": [],
      "activities": [],
      "food": [],
      "stay": ""
    }}
  ],
  "travel_tips": []
}}

RULES:
- No wrapper keys
- No renamed fields
- Start with {{ and end with }}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API. "
                    "Return ONLY valid JSON. "
                    "No markdown. No explanations."
                ),
            },
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.4,
    )

    content = response.choices[0].message.content.strip()

    try:
        itinerary = extract_json(content)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse JSON from OpenAI\n\nRaw output:\n{content}"
        ) from e

    # Final sanity check
    required_keys = {
        "destination",
        "duration",
        "budget_level",
        "activities",
        "days",
        "travel_tips",
    }

    missing = required_keys - itinerary.keys()
    if missing:
        raise RuntimeError(f"Missing required keys: {missing}")

    return itinerary

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Travel Itinerary Generator API",
        "version": "1.0",
        "endpoints": {
            "/generate": "POST - Generate travel itinerary",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/generate")
async def create_itinerary(request: ItineraryRequest):
    """
    Generate a travel itinerary based on user preferences
    """
    try:
        itinerary = generate_itinerary(request)
        return {
            "success": True,
            "data": itinerary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate itinerary: {str(e)}"
        )

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Use PORT env variable for deployment platforms (Render, Railway, Heroku)
    # Falls back to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
