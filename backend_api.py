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

OUTPUT FORMAT RULE:
- Output ONLY valid JSON
- JSON MUST MATCH THE SCHEMA EXACTLY
- DO NOT add explanations or text outside JSON

You MUST generate a FULL-DAY itinerary for EACH day.

STRICT RULES (DO NOT VIOLATE):
- Each day MUST start between 7:00 AM – 9:00 AM
- Each day MUST end between 10:00 PM – 12:00 PM
- Time must progress continuously throughout the day
- NO large gaps longer than 90 minutes unless explicitly stated as rest

FOOD RULES (MANDATORY):
Each day MUST include ALL of the following meals in the "food" array:
- Breakfast (7:00 AM – 9:00 AM)
- Lunch (12:00 PM – 2:30 PM)
- Snacks / Tea (4:00 PM – 6:00 PM) Optional
- Dinner (7:00 PM – 10:00 PM)

If any meal is missing, the response is INVALID.

ACTIVITIES RULES:
- Include morning, afternoon, evening, and night activities
- Use realistic travel and rest gaps
- Every place/activity MUST include:
  start_time, end_time, and duration

    "food": [
  {
    "meal_type": "Breakfast | Lunch | Snacks | Dinner",
    "time": "string",
    "restaurant_name": "string",
    "location": "string",
    "dishes": [
      {
        "name": "string",
        "description": "string"
      }
    ],
    "price_range": "$ | $$ | $$$",
    "why_recommended": "string"
  }
]

    "stay": "Hotel/accommodation recommendation with brief description"
    "travel_tips": [
  "tip1",
  "tip2",
  "tip3",
  "tip4",
  "tip5"
]
    TRAVEL TIPS RULE:
"travel_tips" must be a flexible-length array.
Include as many useful, destination-specific tips as applicable.
There is NO upper limit on the number of tips.
Minimum: 1 tip.


CRITICAL INSTRUCTIONS FOR FOOD RECOMMENDATIONS:
- NEVER use generic terms like "local seafood", "traditional cuisine", "regional dishes"
- ALWAYS specify EXACT dish names (e.g., "Chicken Xacuti", "Prawn Balchão", "Fish Curry Rice", "Bebinca")
- For each meal, recommend 2-3 SPECIFIC dishes by their actual names
- Include REAL restaurant names where possible (well-known local spots)
- Each dish should have a brief description of what it is
- Price range: $ (budget: under $10), $$ (mid-range: $10-30), $$$ (expensive: over $30)

CRITICAL INSTRUCTIONS FOR TIMING:
- Provide SPECIFIC start and end times for every place and activity
- Use 12-hour format (e.g., "9:00 AM", "2:30 PM")
- Times must flow logically throughout the day
- Account for travel time between locations
- Follow the specified pace: 
  * Relaxed: More time at each place, fewer activities
  * Balanced: Mix of time and activities
  * Fast-paced: Many activities, shorter durations
- Start times should typically be between 7:00 AM and 8:00 PM
- Include realistic travel/buffer time

EXAMPLE OF GOOD FOOD ENTRY:
{{
  "meal_type": "Lunch",
  "time": "1:00 PM",
  "restaurant_name": "Fisherman's Wharf",
  "location": "Calangute Beach",
  "dishes": [
    {{
      "name": "Goan Fish Curry Rice",
      "description": "Fresh kingfish in coconut-based curry with steamed rice"
    }},
    {{
      "name": "Prawn Balchão",
      "description": "Spicy tangy prawns in Goan pickle-style gravy"
    }},
    {{
      "name": "Chicken Cafreal",
      "description": "Green masala marinated grilled chicken"
    }}
  ],
  "price_range": "$$",
  "why_recommended": "Authentic Goan cuisine with fresh catch of the day, popular with locals"
}}

EXAMPLE OF GOOD PLACE ENTRY:
{{
  "name": "Chapora Fort",
  "start_time": "9:00 AM",
  "end_time": "10:30 AM",
  "duration": "1.5 hours",
  "description": "Historic Portuguese fort with panoramic views of Vagator Beach and Chapora River. Famous from Bollywood movie Dil Chahta Hai."
}}

RULES:
- No wrapper keys
- No renamed fields
- Start with {{ and end with }}
- ALL times must be specified with start_time and end_time
- ALL dishes must have specific names, NOT generic descriptions
- Include actual restaurant names whenever possible
"""

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