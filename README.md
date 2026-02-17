# 🧭 TrailBuddy — AI Travel Itinerary Generator

> Turn any Instagram travel reel into a personalized, day-by-day itinerary in seconds.

---

## ✨ What It Does

TrailBuddy lets you paste a public Instagram Reel URL and automatically extracts places, landmarks, and activities mentioned in the caption and hashtags. You then pick the spots you actually want to visit, and the AI builds a complete travel itinerary tailored to your selection.

It also supports **manual entry** — just type in a destination and your preferences to generate a custom plan from scratch.

---

## 🗺️ User Flow (Instagram Reel Mode)

```
Step 1 → Paste Reel URL → Extract Places
Step 2 → Select the places you want to visit
Step 3 → AI generates a full day-by-day itinerary
```

No redundant API calls — reel data is fetched **once** in Step 1 and cached through to Step 3.

---

## 🏗️ Architecture

```
Frontend (index.html)
    │
    ├── POST /extract-places-from-reel   ← Step 1: Apify fetches reel, GPT-4o extracts places
    │         returns: places list + cached_reel_data
    │
    └── POST /generate-from-reel         ← Step 2: Uses cached data + selected places → GPT-4o itinerary
              skips Apify (no re-fetch)

    POST /generate                        ← Manual mode: GPT-4o-mini itinerary from form input
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JS |
| Backend | Python, FastAPI, Uvicorn |
| Reel Extraction | [Apify](https://apify.com) — `apify/instagram-scraper` actor |
| AI / LLM | OpenAI `gpt-4o` (reel mode), `gpt-4o-mini` (manual mode) |
| Deployment | Render.com (backend), Netlify (frontend) |

---

## 📁 Project Structure

```
trailbuddy/
├── backend_api.py      # FastAPI backend — all endpoints and AI logic
├── index.html          # Frontend — single-file UI
├── style.css           # Styles (also embedded in index.html)
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- An [Apify API token](https://console.apify.com/account/integrations) (for Instagram reel mode)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/trailbuddy.git
cd trailbuddy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
APIFY_API_TOKEN=apify_api_...
```

### 4. Run the backend

```bash
uvicorn backend_api:app --reload --port 8000
```

### 5. Run the frontend

```bash
python -m http.server 3000
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

> **Note:** Update the `API_URL` constant at the top of the `<script>` in `index.html` to point to your backend:
> ```js
> const API_URL = 'http://localhost:8000';
> ```

---

## 🔌 API Endpoints

### `GET /`
Returns API info and available endpoints.

### `GET /health`
Health check. Confirms whether OpenAI and Apify are configured.

```json
{
  "status": "healthy",
  "openai_configured": true,
  "apify_configured": true
}
```

---

### `POST /extract-places-from-reel`
**Step 1 of the reel flow.** Fetches reel data via Apify, then uses GPT-4o to extract places.

**Request:**
```json
{
  "reel_url": "https://www.instagram.com/reel/ABC123/"
}
```

**Response:**
```json
{
  "success": true,
  "destination": "Rishikesh",
  "country": "India",
  "places": [
    {
      "name": "Laxman Jhula",
      "type": "attraction",
      "description": "Iconic suspension bridge over the Ganges",
      "inferred": false
    }
  ],
  "cached_reel_data": { ... }
}
```

---

### `POST /generate-from-reel`
**Step 2 of the reel flow.** Generates itinerary from cached reel data + user-selected places. Skips Apify entirely if `cached_reel_data` is provided.

**Request:**
```json
{
  "reel_url": "https://www.instagram.com/reel/ABC123/",
  "selected_places": ["Laxman Jhula", "Cafe de Goa", "Triveni Ghat"],
  "cached_reel_data": { ... },
  "duration_override": 3,
  "budget_level": "mid-range"
}
```

---

### `POST /generate`
Manual mode. Generates a full itinerary from user-provided preferences.

**Request:**
```json
{
  "place": "Goa, India",
  "duration": 4,
  "theme": ["Beach", "Foodie"],
  "number_of_people": 2,
  "budget_level": "mid-range",
  "interests": ["Photography", "Sunsets"],
  "pace": "balanced",
  "accommodation_area": "North Goa",
  "transport_preference": "Scooter",
  "food_preference": "Seafood",
  "constraints": [],
  "season_dates": ["March 2026"]
}
```

---

### `POST /debug/extract-reel`
Debug only. Returns raw Apify reel data without generating an itinerary.

---

## 🚀 Deployment

### Backend — Render.com

1. Push code to GitHub
2. Create a new **Web Service** on Render, connect your repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the Render dashboard:
   - `OPENAI_API_KEY`
   - `APIFY_API_TOKEN`

> ⚠️ **Render free tier has a 30-second request timeout.** The Apify scraper can take 30–90s. Upgrade to the Starter plan ($7/mo) or implement async polling to avoid timeout errors.

### Frontend — Netlify

1. Update `API_URL` in `index.html` to your Render backend URL
2. Drag and drop the `index.html` (and `style.css` if separate) into [netlify.com/drop](https://app.netlify.com/drop)

---

## ⚠️ Known Limitations

- **Only public reels** are supported. Private or login-gated reels cannot be scraped.
- **Apify actor changes** — Instagram scraper actors can break when Instagram updates its structure. Monitor the [apify/instagram-scraper](https://apify.com/apify/instagram-scraper) actor page for updates.
- **Render free tier timeouts** — the Apify extraction step can exceed the 30s free-tier limit.
- Itineraries are AI-generated. Always verify restaurant names, timings, and prices before travel.

---

## 📜 Legal & Ethics

- Only public Instagram content is accessed
- No login-based scraping is performed
- Reel data is not stored persistently
- All generated itineraries include the disclaimer: *"AI-generated — verify details before travel"*

---

## 🔮 Roadmap

- [ ] Voice/audio extraction from reels (Whisper)
- [ ] Map auto-routing between selected places
- [ ] Budget calculator per itinerary
- [ ] "Improve this itinerary" edit mode
- [ ] Multi-reel merging → one combined trip
- [ ] Google Maps deep links for every place

---
