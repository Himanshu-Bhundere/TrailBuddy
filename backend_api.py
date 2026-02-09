"""
AI Travel Itinerary Generator - FastAPI Backend
Web API for generating travel itineraries using OpenAI
"""

import os
import json
import traceback
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
    allow_origins=[
        "https://trailbuddy-ai-itinerary.netlify.app",   #Netlify domain
        "http://localhost:8000"
    ],
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
    
    PROMPT = """
Act as an expert travel consultant with 10+ years of experience in luxury and local-authentic travel planning.

Create a highly detailed, optimized travel itinerary based on the following constraints.

DESTINATION: {place}
TRIP DURATION: {duration} days
THEME: {theme}
TRAVELERS: {travelers}
BUDGET LEVEL: {budget}
INTERESTS: {interests}
PACE: {pace}
ACCOMMODATION AREA: {accommodation}
TRANSPORT: {transport}
FOOD: {food}
CONSTRAINTS: {constraints}
SEASON / DATES: {season}

========================
OUTPUT FORMAT (STRICT)
========================
- Output ONLY valid JSON
- JSON MUST start with {{ and end with }}
- DO NOT include markdown, explanations, or comments
- DO NOT include any text outside JSON

========================
TOP-LEVEL JSON KEYS
========================
The JSON object MUST include the following keys:

- destination
- duration
- budget_level
- days
- travel_tips

========================
DAY STRUCTURE RULES
========================
- Generate a FULL-DAY itinerary for EACH day
- Each day MUST start between 7:00 AM – 9:00 AM
- Each day MUST end between 10:00 PM – 12:00 AM
- Time must progress continuously
- No gaps longer than 90 minutes unless explicitly stated as rest

========================
ACTIVITY RULES
========================
- Include morning, afternoon, evening, and night activities
- Every place or activity MUST include:
  - start_time
  - end_time
  - duration

========================
FOOD RULES (MANDATORY)
========================
Each day MUST include ALL meals in the "food" array:
- Breakfast (7:00 AM – 9:00 AM)
- Lunch (12:00 PM – 2:30 PM)
- Snacks / Tea (4:00 PM – 6:00 PM) [optional]
- Dinner (7:00 PM – 10:00 PM)

========================
FOOD OBJECT SCHEMA
========================
Each food item MUST follow this structure:

{{
  "meal_type": "Breakfast | Lunch | Snacks | Dinner",
  "time": "string",
  "restaurant_name": "string",
  "location": "string",
  "dishes": [
    {{
      "name": "string",
      "description": "string"
    }}
  ],
  "price_range": "$ | $$ | $$$",
  "why_recommended": "string"
}}

========================
CRITICAL FOOD INSTRUCTIONS
========================
- NEVER use generic terms like "local food" or "traditional cuisine"
- ALWAYS use REAL dish names (e.g., Chicken Xacuti, Prawn Balchão)
- Recommend 2–3 dishes per meal
- Include REAL restaurant names where possible
- Price range meaning:
  $   = under $10
  $$  = $10–30
  $$$ = over $30

========================
EXAMPLE FOOD ENTRY
========================
{{
  "meal_type": "Lunch",
  "time": "1:00 PM",
  "restaurant_name": "Fisherman's Wharf",
  "location": "Calangute Beach",
  "dishes": [
    {{
      "name": "Goan Fish Curry Rice",
      "description": "Fresh kingfish cooked in coconut-based curry with steamed rice"
    }},
    {{
      "name": "Prawn Balchão",
      "description": "Spicy tangy prawns in traditional Goan pickle-style gravy"
    }},
    {{
      "name": "Chicken Cafreal",
      "description": "Green masala marinated grilled chicken"
    }}
  ],
  "price_range": "$$",
  "why_recommended": "Authentic Goan cuisine popular with locals"
}}

========================
EXAMPLE PLACE ENTRY
========================
{{
  "name": "Chapora Fort",
  "start_time": "9:00 AM",
  "end_time": "10:30 AM",
  "duration": "1.5 hours",
  "description": "Historic Portuguese fort with panoramic views of Vagator Beach and the Chapora River"
}}

========================
TRAVEL TIPS RULES
========================
- travel_tips must be an array
- Minimum 1 tip
- No upper limit
- Tips must be destination-specific

========================
FINAL RULES (DO NOT BREAK)
========================
- No wrapper keys
- No renamed fields
- ALL times must use 12-hour format
- ALL activities must include start_time and end_time
- JSON MUST be valid and parseable
""".format(
    place=request.place,
    duration=request.duration,
    theme=", ".join(request.theme),
    travelers=request.number_of_people,
    budget=request.budget_level,
    interests=", ".join(request.interests),
    pace=request.pace,
    accommodation=request.accommodation_area,
    transport=request.transport_preference,
    food=request.food_preference,
    constraints=", ".join(request.constraints) if request.constraints else "None",
    season=", ".join(request.season_dates) if request.season_dates else "Not specified",
)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API that generates detailed travel itineraries. "
                    "Return ONLY valid JSON with specific dish names and exact timings. "
                    "No markdown. No explanations. No generic food terms."
                ),
            },
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.5,
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
        "days",
        "travel_tips",
    }

    missing = required_keys - itinerary.keys()
    if missing:
        raise RuntimeError(f"Missing required keys: {missing}")
    
    for day in itinerary.get("days", []):
        activities = []
        food = []

        for item in day.get("activities", []):
            if "meal_type" in item:
                food.append(item)
            else:
                activities.append(item)

        day["activities"] = activities
        day["food"] = food


    return itinerary

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Travel Itinerary Generator API",
        "version": "2.0",
        "endpoints": {
            "/generate": "POST - Generate travel itinerary with detailed timings and specific dishes",
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
        return {"success": True, "data": itinerary}
    except Exception as e:
        print("❌ ERROR IN /generate")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)