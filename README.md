# 🌍 TrailBuddy — AI Travel Itinerary Generator

> Paste an Instagram travel reel. Get a personalised day-by-day itinerary in minutes.

TrailBuddy analyses Instagram reels using computer vision, speech recognition, and large language models to extract every place, hotel, restaurant, and activity visible in the video — then turns them into a complete travel plan.

---

## ✨ Features

- **Instagram Reel Analysis** — paste any public travel reel URL and TrailBuddy does the rest
- **Concurrent Video Pipeline** — frames and audio are processed simultaneously via `asyncio`
- **GPT-4o Vision OCR** — reads on-screen text, day labels, prices, hotel names, and itinerary slides at full resolution
- **Whisper Transcription** — extracts spoken place names, tips, and travel cues from the voiceover
- **Smart 4-State Cache** — skips only the parts already stored in the database
- **Manual Mode** — enter destination, duration, themes, and preferences without a reel
- **Real-Time Progress UI** — live step-by-step progress stream via Server-Sent Events (SSE)

---

## 🏗️ Architecture

```
User
 ↓
Paste Instagram Reel URL
 ↓
┌──────────────────────────────────────────┐
│  Cache Check (Supabase)                  │
│  State A: nothing → full pipeline        │
│  State B: caption cached → skip Apify   │
│  State C: video cached → skip video     │
│  State D: both cached → instant places  │
└──────────────────────────────────────────┘
 ↓
Apify → caption + hashtags + video URL
 ↓
Download video (R2 → Instagram CDN fallback)
 ↓
┌─────────────────────┬────────────────────┐
│  ffmpeg             │  ffmpeg            │  ← CONCURRENT
│  Frame extraction   │  Audio extraction  │
└─────────────────────┴────────────────────┘
 ↓
┌─────────────────────┬────────────────────┐
│  GPT-4o Vision      │  Whisper STT       │  ← CONCURRENT
│  OCR + scene data   │  Transcript + cues │
└─────────────────────┴────────────────────┘
 ↓
GPT-4o Consolidation (merge all signals)
 ↓
┌─────────────────────┬────────────────────┐
│  Save → Supabase    │  Upload → R2       │  ← CONCURRENT
└─────────────────────┴────────────────────┘
 ↓
Place extraction (LLM using all data)
 ↓
User selects places
 ↓
GPT-4o Itinerary generation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11 |
| LLM | OpenAI GPT-4o (vision + text) |
| Speech-to-Text | OpenAI Whisper |
| Video Processing | ffmpeg (scene-change + dense frame extraction) |
| Instagram Scraping | Apify (`apify/instagram-scraper`) |
| Database | Supabase (PostgreSQL) |
| Video Storage | Cloudflare R2 (S3-compatible) |
| Frontend | Vanilla HTML/CSS/JS with SSE streaming |
| Deployment | Render.com (backend) + Netlify (frontend) |

---

## 📁 Project Structure

```
TrailBuddy/
├── backend_api.py       # FastAPI app — SSE pipeline, all endpoints
├── video_processor.py   # Video intelligence — frames, OCR, Whisper, consolidation
├── storage_backend.py   # Supabase + R2 storage abstraction
├── index.html           # Frontend — SSE progress UI + itinerary display
├── style.css            # Styles
├── render.yaml          # Render.com deployment config (includes ffmpeg)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Apify (Instagram scraping)
APIFY_API_TOKEN=apify_api_...

# Storage backend — "supabase_r2" for production, "local" for development
STORAGE_BACKEND=supabase_r2

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Cloudflare R2
R2_ACCOUNT_ID=abc123
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=trailbuddy-videos
R2_ENDPOINT_URL=https://abc123.r2.cloudflarestorage.com
R2_PUBLIC_URL=https://pub-xxx.r2.dev   # optional — for public bucket access
```

---

## 🚀 Local Development

### Prerequisites

- Python 3.11+
- ffmpeg installed at OS level

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Himanshu-Bhundere/TrailBuddy
cd TrailBuddy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env (see Environment Variables above)
cp .env.example .env

# 4. Run the backend
uvicorn backend_api:app --reload --port 8000
```

### Frontend

The frontend is a single `index.html` file — open it directly in a browser or serve it with any static server:

```bash
# Quick static server
python -m http.server 3000
# Open http://localhost:3000
```

> Make sure `API_URL` in `index.html` points to `http://localhost:8000` for local development.

---

## 🗄️ Database Setup

Run the following SQL in your Supabase SQL Editor:

```sql
CREATE TABLE IF NOT EXISTS reel_cache (
    id                      BIGSERIAL   PRIMARY KEY,
    reel_id                 TEXT        UNIQUE NOT NULL,
    url                     TEXT,
    caption                 TEXT,
    hashtags                TEXT[]      DEFAULT '{}',
    location                TEXT,
    likes                   INTEGER     DEFAULT 0,
    timestamp               TEXT,
    owner_username          TEXT,
    video_url               TEXT,
    display_url             TEXT,
    r2_video_key            TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Add video intelligence columns (safe to run even if table exists)
ALTER TABLE reel_cache ADD COLUMN IF NOT EXISTS video_processed        BOOLEAN DEFAULT FALSE;
ALTER TABLE reel_cache ADD COLUMN IF NOT EXISTS video_insights         JSONB;
ALTER TABLE reel_cache ADD COLUMN IF NOT EXISTS inferred_duration_days INTEGER;
ALTER TABLE reel_cache ADD COLUMN IF NOT EXISTS inferred_budget_level  TEXT;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reel_cache_reel_id         ON reel_cache (reel_id);
CREATE INDEX IF NOT EXISTS idx_reel_cache_video_processed ON reel_cache (video_processed);
CREATE INDEX IF NOT EXISTS idx_reel_cache_created_at      ON reel_cache (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reel_cache_vi_gin          ON reel_cache USING GIN (video_insights);
```

---

## 🌐 Deployment (Render.com)

The `render.yaml` in the repo root handles everything:

```yaml
services:
  - type: web
    name: trailbuddy
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.8
    packages:
      - ffmpeg          # installed as a system package before pip runs
```

**Steps:**
1. Push code to GitHub
2. Create a new Web Service on Render → connect your repo
3. Render auto-detects `render.yaml`
4. Add all environment variables in the Render dashboard → Environment tab
5. Deploy

> The `packages: [ffmpeg]` declaration installs ffmpeg system-wide before pip runs. Do **not** use `apt-get install` in the build command — Render runs as non-root.

**Frontend (Netlify):**
1. Drag and drop `index.html` + `style.css` into Netlify
2. Make sure `API_URL` in `index.html` points to your Render URL

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze-reel` | SSE stream — full concurrent reel analysis pipeline |
| `POST` | `/generate-from-reel` | Generate itinerary from analysed reel data |
| `POST` | `/generate` | Generate itinerary from manual form input |
| `GET` | `/cache-status?reel_url=...` | Debug — show what is cached for a reel |
| `GET` | `/health` | Health check |
| `POST` | `/extract-places-from-reel` | Legacy endpoint (backward compatibility) |

### `/analyze-reel` Request

```json
{
  "reel_url":   "https://www.instagram.com/reel/ABC123/",
  "skip_audio": false,
  "skip_video": false
}
```

### SSE Event Format

The endpoint streams `text/event-stream` events as each pipeline stage completes:

```
data: {"step":"apify","status":"done","message":"18 hashtags · \"Goa trip…\"","progress":28}

data: {"step":"vision","status":"running","message":"GPT-4o analysing 45 frames…","progress":55}

data: {"step":"complete","status":"done","message":"Ready!","progress":100,
       "destination":"Goa","country":"India","places":[...],"video_insights":{...}}
```

**Step names:** `cache_check` → `apify` → `download` → `frames` → `audio_extract` → `vision` → `whisper` → `consolidate` → `save` → `places` → `complete`

**Status values:** `idle` | `running` | `done` | `skipped` | `failed`

---

## 🔍 Cache Debug

Check what is stored for any reel without running the full pipeline:

```
GET https://trailbuddy.onrender.com/cache-status?reel_url=https://instagram.com/reel/ABC123/
```

Response:
```json
{
  "reel_id": "ABC123",
  "cached": true,
  "state": "D — FULL HIT: caption + video both cached → places shown instantly",
  "fields": {
    "caption": true,
    "hashtags_count": 18,
    "video_processed": true,
    "video_insights_present": true,
    "video_insights_places": 13,
    "video_insights_vibe": "backpacking",
    "inferred_duration_days": 5,
    "r2_video_key": "ABC123.mp4"
  }
}
```

---

## 🎬 Video Intelligence Pipeline

TrailBuddy uses a two-pass frame extraction strategy to maximise OCR accuracy on travel reels:

**Pass A — Scene-change detection** (`ffmpeg select filter, threshold 0.25`):
Outputs a frame whenever the image changes significantly. Catches every text slide, location card, and itinerary overlay regardless of how briefly it appears.

**Pass B — Dense uniform sampling** (`0.5s interval = 2 fps`):
Fills gaps left by the scene filter for slow-zoom shots where text fades in gradually.

Frames from both passes are merged, deduplicated (consecutive frames within 3% file-size = same content), and capped at 60 unique frames. All frames are sent to GPT-4o Vision at `detail="high"` in parallel batches of 15.

The vision prompt has an explicit `itinerary_slides` field that captures the full text of any day-plan card or schedule visible in the video.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/map-view`
3. Commit your changes: `git commit -m 'Add map-based itinerary view'`
4. Push to the branch: `git push origin feature/map-view`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by [Himanshu Bhundere](https://github.com/Himanshu-Bhundere)