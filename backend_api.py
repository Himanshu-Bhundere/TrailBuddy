"""
Unified Travel Itinerary Generator - FastAPI Backend
Supports both manual input and Instagram Reel URL extraction
"""

import os
import json
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from apify_client import ApifyClient

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")  # Optional for manual mode

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
apify_client = ApifyClient(APIFY_API_TOKEN) if APIFY_API_TOKEN else None

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(title="Unified Travel Itinerary Generator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trailbuddy-ai-itinerary.netlify.app",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",  # For Live Server
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ManualItineraryRequest(BaseModel):
    """Request model for manual itinerary generation"""
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

class ReelRequest(BaseModel):
    """Request model for Instagram reel-based generation"""
    reel_url: str
    duration_override: Optional[int] = None
    budget_level: Optional[str] = None

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def extract_json(text: str) -> dict:
    """Safely extract JSON object from LLM output"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in OpenAI response")
    return json.loads(text[start:end + 1])

# --------------------------------------------------
# MANUAL ITINERARY GENERATION
# --------------------------------------------------

def generate_manual_itinerary(request: ManualItineraryRequest) -> dict:
    """Generate travel itinerary from manual user input"""
    
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

    response = openai_client.chat.completions.create(
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
    itinerary = extract_json(content)

    # Validate required keys
    required_keys = {"destination", "duration", "budget_level", "days", "travel_tips"}
    missing = required_keys - itinerary.keys()
    if missing:
        raise RuntimeError(f"Missing required keys: {missing}")

    # Separate activities and food
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
# INSTAGRAM REEL EXTRACTION
# --------------------------------------------------

def extract_reel_data(reel_url: str) -> dict:
    """Extract caption and hashtags from Instagram reel using Apify"""
    
    if not apify_client:
        raise HTTPException(
            status_code=503,
            detail="Apify integration not configured. Please add APIFY_API_TOKEN to .env file."
        )
    
    print(f"🔍 Extracting data from: {reel_url}")
    
    run_input = {
        "directUrls": [reel_url],
        "resultsType": "posts",
        "resultsLimit": 1,
        "searchType": "hashtag",
        "searchLimit": 1,
    }
    
    try:
        run = apify_client.actor("apify/instagram-scraper").call(run_input=run_input)
        items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if not items:
            raise ValueError("No data extracted from Instagram reel")
        
        reel_data = items[0]
        
        extracted = {
            "url": reel_url,
            "caption": reel_data.get("caption", ""),
            "hashtags": reel_data.get("hashtags", []),
            "location": reel_data.get("locationName", ""),
            "likes": reel_data.get("likesCount", 0),
            "raw_data": reel_data
        }
        
        print(f"✅ Extracted caption: {extracted['caption'][:100]}...")
        print(f"✅ Found {len(extracted['hashtags'])} hashtags")
        
        return extracted
        
    except Exception as e:
        print(f"❌ Apify extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract reel data: {str(e)}")

# --------------------------------------------------
# INSTAGRAM REEL ITINERARY GENERATION
# --------------------------------------------------

def generate_reel_itinerary(reel_data: dict, duration_override: Optional[int] = None, budget_level: Optional[str] = None) -> dict:
    """Generate travel itinerary from Instagram caption and hashtags"""
    
    caption = reel_data.get("caption", "")
    hashtags = reel_data.get("hashtags", [])
    location = reel_data.get("location", "")
    
    hashtag_text = " ".join([f"#{tag}" for tag in hashtags])
    full_text = f"{caption}\n\nHashtags: {hashtag_text}"
    if location:
        full_text += f"\n\nLocation tag: {location}"
    
    PROMPT = f"""
You are a travel expert analyzing an Instagram Reel to extract a detailed travel itinerary.

INSTAGRAM REEL DATA:
{full_text}

TASK:
Extract and infer the following information:
1. Destination (city/region/country)
2. Trip duration (in days)
3. Travel vibe/theme (adventure, luxury, backpacking, food tour, etc.)
4. Key activities mentioned or implied
5. Budget level (if not mentioned, infer from context)

ADDITIONAL CONSTRAINTS:
{f"- User specified trip duration: {duration_override} days" if duration_override else "- Infer trip duration from content"}
{f"- User specified budget level: {budget_level}" if budget_level else "- Infer budget level from content"}

OUTPUT REQUIREMENTS:
Return ONLY a valid JSON object (no markdown, no explanations) with this exact structure:

{{
  "destination": "string (city or region)",
  "country": "string",
  "duration": number,
  "budget_level": "budget | mid-range | luxury",
  "theme": ["array of themes"],
  "vibe": "string (overall trip vibe)",
  "days": [
    {{
      "day": number,
      "title": "string (e.g., 'Beach Hopping & Sunset Views')",
      "activities": [
        {{
          "name": "string",
          "start_time": "string (e.g., '9:00 AM')",
          "end_time": "string (e.g., '11:00 AM')",
          "duration": "string (e.g., '2 hours')",
          "description": "string",
          "category": "sightseeing | food | adventure | relaxation | culture"
        }}
      ],
      "food": [
        {{
          "meal_type": "Breakfast | Lunch | Dinner | Snacks",
          "time": "string",
          "restaurant_name": "string (or 'Local eatery' if unknown)",
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
      ],
      "accommodation": {{
        "type": "hotel | hostel | resort | homestay | camping",
        "area": "string (neighborhood or region)",
        "suggestion": "string"
      }}
    }}
  ],
  "key_highlights": ["array of main attractions/experiences"],
  "estimated_budget": {{
    "total": "string (e.g., '$500-800 per person')",
    "breakdown": {{
      "accommodation": "string",
      "food": "string",
      "activities": "string",
      "transport": "string"
    }}
  }},
  "travel_tips": ["array of practical tips"],
  "best_time_to_visit": "string",
  "packing_suggestions": ["array of items to pack"]
}}

IMPORTANT RULES:
- If information is not explicitly mentioned, make reasonable inferences based on:
  * Hashtags (e.g., #budgettravel, #luxurytravel, #3daysInBali)
  * Mentioned activities (e.g., "rafting" suggests adventure)
  * Location context (e.g., Goa = beach, Ladakh = mountains)
  * Visual cues from caption (e.g., "stayed at luxury resort")
  
- Generate a COMPLETE day-by-day itinerary even if the reel only shows highlights
- Use actual dish names when food is mentioned
- Suggest realistic timings (start days at 8-9 AM, end at 9-10 PM)
- Include all meals (breakfast, lunch, dinner) for each day
- Make the itinerary actionable and practical
- ALL activities must include start_time, end_time, and duration

Begin analysis:
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a travel planning expert who analyzes Instagram content and creates detailed itineraries. "
                        "Return ONLY valid JSON. No markdown. No explanations outside the JSON structure."
                    )
                },
                {"role": "user", "content": PROMPT}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content.strip()
        itinerary = extract_json(content)
        
        # Validate required fields
        required_keys = {"destination", "duration", "budget_level", "days"}
        missing = required_keys - itinerary.keys()
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        
        print(f"✅ Generated itinerary for {itinerary['destination']} ({itinerary['duration']} days)")
        
        return itinerary
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {str(e)}")
        print(f"Raw response: {content}")
        raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON")
    except Exception as e:
        print(f"❌ LLM generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {str(e)}")

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Unified Travel Itinerary Generator API",
        "version": "3.0",
        "description": "Generate itineraries from manual input or Instagram Reels",
        "endpoints": {
            "/generate": "POST - Generate itinerary from manual input",
            "/generate-from-reel": "POST - Generate itinerary from Instagram reel URL",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "apify_configured": bool(APIFY_API_TOKEN),
        "features": {
            "manual_generation": True,
            "instagram_generation": bool(APIFY_API_TOKEN)
        }
    }

@app.post("/generate")
async def create_manual_itinerary(request: ManualItineraryRequest):
    """Generate a travel itinerary from manual user input"""
    try:
        itinerary = generate_manual_itinerary(request)
        return {"success": True, "data": itinerary, "source": "manual"}
    except Exception as e:
        print("❌ ERROR IN /generate")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-from-reel")
async def create_reel_itinerary(request: ReelRequest):
    """Generate a travel itinerary from Instagram reel URL"""
    try:
        # Step 1: Extract reel data
        reel_data = extract_reel_data(request.reel_url)
        
        # Step 2: Generate itinerary
        itinerary = generate_reel_itinerary(
            reel_data,
            duration_override=request.duration_override,
            budget_level=request.budget_level
        )
        
        # Step 3: Return combined result
        return {
            "success": True,
            "source": "instagram",
            "reel_data": {
                "url": reel_data["url"],
                "caption": reel_data["caption"],
                "hashtags": reel_data["hashtags"],
                "location": reel_data["location"]
            },
            "data": itinerary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print("❌ ERROR IN /generate-from-reel")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/extract-reel")
async def debug_extract_reel(request: ReelRequest):
    """Debug endpoint: Only extract reel data without generating itinerary"""
    try:
        reel_data = extract_reel_data(request.reel_url)
        return {"success": True, "data": reel_data}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)