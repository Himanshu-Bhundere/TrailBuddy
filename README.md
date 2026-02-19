# 🧭 TrailBuddy — AI Travel Itinerary Generator

> Turn any Instagram travel reel into a personalized, day-by-day itinerary in seconds.

---

## ✨ Features

- **🎥 Instagram Reel → Itinerary**: Paste a reel URL, extract places, and generate a full travel plan
- **📍 Smart Place Extraction**: AI identifies cities, attractions, restaurants from captions & hashtags
- **🎯 Customizable**: Select which places you want to visit before generating
- **💾 Persistent Cache**: Supabase + Cloudflare R2 storage (same reel = instant for all users)
- **✍️ Manual Mode**: Create itineraries from scratch without reels
- **🌐 Production Ready**: Zero-downtime deployment on Render.com

---

## 🗺️ User Flow

```
Step 1 → Paste Reel URL → AI extracts places
Step 2 → Select places you want to visit
Step 3 → Generate detailed day-by-day itinerary
```

---

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS)
    ↓
FastAPI Backend
    ↓
├─ Apify → Instagram reel data
├─ OpenAI GPT-4o → Extract places & generate itinerary
└─ Supabase + R2 → Cache (metadata + videos)
```

**Storage:**
- **Supabase** (PostgreSQL): JSON metadata, queryable cache
- **Cloudflare R2**: Video files, zero egress fees

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JS |
| Backend | Python, FastAPI, Uvicorn |
| Reel Scraping | Apify Instagram Scraper |
| AI | OpenAI GPT-4o (reel), GPT-4o-mini (manual) |
| Cache Storage | Supabase + Cloudflare R2 |
| Deployment | Render.com (backend), Netlify (frontend) |

---

## 📁 Project Structure

```
trailbuddy/
├── backend_api.py           # FastAPI backend
├── storage_backend.py       # Supabase + R2 storage layer
├── index.html               # Frontend UI
├── style.css               # Styles
├── requirements.txt         # Python dependencies
└── .env.example            # Environment template
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Apify token](https://console.apify.com/account/integrations)
- [Supabase account](https://supabase.com) (free)
- [Cloudflare account](https://cloudflare.com) with R2 enabled (free)

---

### 1. Clone & Install

```bash
git clone https://github.com/your-username/trailbuddy.git
cd trailbuddy
pip install -r requirements.txt
```

---

### 2. Set Up Supabase

1. Create project at [supabase.com](https://supabase.com)
2. Run this SQL in SQL Editor:

```sql
CREATE TABLE reel_cache (
    reel_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    caption TEXT,
    hashtags TEXT[] DEFAULT '{}',
    location TEXT,
    likes INTEGER DEFAULT 0,
    timestamp TEXT,
    owner_username TEXT,
    video_url TEXT,
    display_url TEXT,
    r2_video_key TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

3. Get credentials:
   - Settings → API → **Project URL**
   - Settings → API → **`service_role` key** (secret)

---

### 3. Set Up Cloudflare R2

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → R2
2. Create bucket: `trailbuddy-reels`
3. Manage API Tokens → Create token:
   - Permissions: **Object Read & Write**
   - Bucket: `trailbuddy-reels`
4. Copy: **Account ID**, **Access Key ID**, **Secret Access Key**

---

### 4. Configure Environment

Create `.env` file:

```env
# OpenAI & Apify
OPENAI_API_KEY=sk-...
APIFY_API_TOKEN=apify_api_...

# Storage
STORAGE_BACKEND=supabase_r2

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGci...

# Cloudflare R2
R2_ACCOUNT_ID=a1b2c3d4
R2_ACCESS_KEY_ID=abc123
R2_SECRET_ACCESS_KEY=xyz789
R2_BUCKET_NAME=trailbuddy-reels
R2_ENDPOINT_URL=https://a1b2c3d4.r2.cloudflarestorage.com
```

---

### 5. Run Locally

```bash
# Start backend
uvicorn backend_api:app --reload --port 8000

# In another terminal, start frontend
python -m http.server 3000
```

Open [http://localhost:3000](http://localhost:3000)

---

## 🚀 Deploy to Production

### Backend (Render.com)

1. Push to GitHub
2. Create Web Service on [render.com](https://render.com)
3. Connect repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
6. Add environment variables (from `.env` above)
7. Deploy ✅

### Frontend (Netlify)

1. Update `API_URL` in `index.html`:
   ```javascript
   const API_URL = 'https://your-app.onrender.com';
   ```
2. Drag `index.html` to [netlify.com/drop](https://app.netlify.com/drop)
3. Done ✅

---

## 📊 Free Tier Limits

| Service | Free Tier | Enough For |
|---------|-----------|------------|
| **Supabase** | 500 MB database | ~250,000 reel metadata entries |
| **Cloudflare R2** | 10 GB storage | ~2,000 videos (5 MB each) |
| **R2 Egress** | **Unlimited FREE** | 🎉 No bandwidth charges |
| **Render.com** | 750 hours/month | ~1 month uptime |

---

## 🔍 API Endpoints

### `GET /health`
Health check with configuration status.

### `POST /extract-places-from-reel`
Extract places from Instagram reel URL.

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
  "destination": "Bali",
  "places": [
    {
      "name": "Ubud Rice Terraces",
      "type": "attraction",
      "description": "UNESCO heritage site"
    }
  ]
}
```

### `POST /generate-from-reel`
Generate itinerary from selected places.

**Request:**
```json
{
  "reel_url": "https://www.instagram.com/reel/ABC123/",
  "selected_places": ["Ubud Rice Terraces", "Tanah Lot Temple"],
  "duration_override": 3,
  "budget_level": "mid-range"
}
```

### `POST /generate`
Manual itinerary generation (no reel).

---

## 💡 How Caching Works

**First Request (Cache Miss):**
```
User → Backend → Apify (30-90s) → Download video
                    ↓
           Upload to R2 + Save to Supabase
                    ↓
              Return itinerary
```

**Second Request (Cache Hit):**
```
User → Backend → Check Supabase → Found! (<1s)
                    ↓
              Return cached data
```

**Result:** 99% faster for repeat requests 🚀

---

## 🐛 Troubleshooting

### "Missing Supabase config"
- Check `.env` has `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`
- Use **service_role** key (not anon)

### "relation 'reel_cache' does not exist"
- Run SQL from Step 2 in Supabase SQL Editor

### "Access Denied" (R2)
- Verify API token permissions: "Object Read & Write"
- Check `R2_BUCKET_NAME` matches exactly

### Render timeout (30s limit)
- Upgrade to Starter plan ($7/mo)
- Or implement async polling

---

## 📚 Documentation

- [Supabase Setup Guide](./SUPABASE_R2_SETUP.md) — Full setup walkthrough
- [Quick Start](./SUPABASE_R2_QUICKSTART.md) — 5-minute setup

---

## 🔐 Security

- Never commit `.env` to Git
- Use environment variables in production
- Rotate API keys periodically
- Supabase `service_role` key bypasses RLS — keep it secret

---

## 📈 Roadmap

- [ ] Voice/audio extraction (Whisper)
- [ ] Map routing between places
- [ ] Budget calculator
- [ ] Multi-reel merging
- [ ] Google Maps integration
- [ ] Export to PDF/Google Calendar

---

## 📄 License

MIT License — Free to use, modify, and deploy.

---

## 🎯 Quick Links

- **Live Demo**: [trailbuddy.netlify.app](https://trailbuddy-ai-itinerary.netlify.app/)
- **Supabase**: [supabase.com](https://supabase.com)
- **Cloudflare R2**: [cloudflare.com/r2](https://www.cloudflare.com/products/r2/)
- **OpenAI**: [platform.openai.com](https://platform.openai.com)
- **Apify**: [apify.com](https://apify.com)

---

Built with ❤️ for travelers who want instant, AI-powered itineraries.