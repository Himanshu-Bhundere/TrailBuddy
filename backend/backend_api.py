"""
backend_api.py — TrailBuddy Unified Backend
=============================================
Key change: Instagram reel analysis is now a SINGLE streaming endpoint
(/analyze-reel) that runs the entire pipeline concurrently and pushes
real-time progress events via Server-Sent Events (SSE).

Flow inside /analyze-reel:
  1. Cache check         (instant)
  2. Apify fetch         (15–30 s, sequential — we need caption + video_url first)
  3. Video download      (5–20 s, sequential — needs video_url from Apify)
      ┌─────────────────────────────────────────────────────┐
  4.  │ Frame extraction (ffmpeg)  │  Audio extraction (ffmpeg) │  ← CONCURRENT
      └─────────────────────────────────────────────────────┘
      ┌─────────────────────────────────────────────────────┐
  5.  │ GPT-4o Vision              │  Whisper STT               │  ← CONCURRENT
      └─────────────────────────────────────────────────────┘
  6. Consolidation       (LLM merge of all signals)
      ┌─────────────────────────────────────────────────────┐
  7.  │ Save metadata → Supabase  │  Upload video → R2         │  ← CONCURRENT
      └─────────────────────────────────────────────────────┘
  8. Place extraction    (LLM using ALL consolidated data)
  9. Complete event      (returns everything to frontend)
"""

import os
import json
import asyncio
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, AsyncGenerator

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from apify_client import ApifyClient

from storage_backend import (
    get_storage_backend,
    extract_reel_id_from_url,
    download_video,
    StorageBackend,
)
from video_processor import (
    extract_frames,
    extract_audio,
    analyse_frames_with_vision,
    transcribe_audio,
    consolidate_video_insights,
    extract_places_from_all_data,
    cleanup_temp_files,
    ffmpeg_available,
)

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
apify_client  = ApifyClient(APIFY_API_TOKEN) if APIFY_API_TOKEN else None

try:
    storage: StorageBackend = get_storage_backend()
    print(f"✅ Storage: {storage.__class__.__name__}")
except Exception as e:
    print(f"⚠️  Storage init failed: {e}")
    storage = None

# Thread pool for blocking I/O run inside asyncio
_executor = ThreadPoolExecutor(max_workers=6)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TrailBuddy — Unified Travel Itinerary API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────

class ManualItineraryRequest(BaseModel):
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

class AnalyzeReelRequest(BaseModel):
    """Single request model — replaces the old two-step approach."""
    reel_url: str
    skip_audio: bool = False   # True = skip Whisper (faster but loses voice data)
    skip_video: bool = False   # True = skip all video processing (caption-only mode)

class GenerateFromReelRequest(BaseModel):
    reel_url: str
    selected_places: List[str] = []
    duration_override: Optional[int] = None
    budget_level: Optional[str] = None
    # Passed from frontend after /analyze-reel completes
    cached_reel_data: Optional[dict] = None
    video_insights: Optional[dict] = None

# ── Utilities ─────────────────────────────────────────────────────────────────

def _json(text: str) -> dict:
    """Extract JSON from LLM output (strips markdown fences)."""
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])

async def _run(fn, *args):
    """Run a blocking function in the thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)

def _sse(step: str, status: str, message: str, progress: int, **extra) -> str:
    """Format a Server-Sent Event line."""
    payload = {"step": step, "status": status, "message": message, "progress": progress}
    payload.update(extra)
    return f"data: {json.dumps(payload)}\n\n"


# ═══════════════════════════════════════════════════════════════════════════
# APIFY HELPERS  (sync — called via _run())
# ═══════════════════════════════════════════════════════════════════════════

def _apify_fetch(reel_url: str) -> dict:
    """Fetch reel metadata from Apify. Returns extracted dict."""
    if not apify_client:
        raise RuntimeError("Apify not configured — add APIFY_API_TOKEN to .env")

    run = apify_client.actor("apify/instagram-scraper").call(run_input={
        "directUrls": [reel_url],
        "resultsType": "posts",
        "resultsLimit": 1,
        "searchType": "hashtag",
        "searchLimit": 1,
    })

    items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        raise ValueError("Apify returned no data for this reel URL")

    d = items[0]
    return {
        "url":           reel_url,
        "caption":       d.get("caption", ""),
        "hashtags":      d.get("hashtags", []),
        "location":      d.get("locationName", ""),
        "likes":         d.get("likesCount", 0),
        "timestamp":     d.get("timestamp"),
        "owner_username": d.get("ownerUsername", ""),
        "video_url":     d.get("videoUrl", ""),
        "display_url":   d.get("displayUrl", ""),
    }


def _download_video(video_url: str) -> Optional[str]:
    """Download video from CDN URL to a temp file. Returns local path or None."""
    if not video_url:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="tb_reel_")
        path = tmp.name
        tmp.close()
        if download_video(video_url, path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ Video downloaded: {size_mb:.2f} MB → {path}")
            return path
        return None
    except Exception as e:
        print(f"❌ Video download failed: {e}")
        return None


def _download_video_from_r2(r2_key: str) -> Optional[str]:
    """
    Download video directly from Cloudflare R2 (already uploaded on a previous run).
    Avoids re-hitting the Instagram CDN, which expires URLs and has rate limits.
    Returns local temp path or None on failure.
    """
    if not storage or not hasattr(storage, "s3") or not r2_key:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="tb_reel_r2_")
        path = tmp.name
        tmp.close()

        print(f"⬇️  Downloading from R2: {r2_key} → {path}")
        storage.s3.download_file(storage.r2_bucket, r2_key, path)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ R2 video loaded: {size_mb:.2f} MB → {path}")
        return path
    except Exception as e:
        print(f"⚠️  R2 download failed ({r2_key}): {e} — will fall back to CDN")
        return None


def _save_to_supabase(reel_id: str, reel_data: dict, insights: dict) -> None:
    """Save reel metadata + VideoInsights to Supabase."""
    if not storage:
        return
    try:
        # Upsert base metadata (r2_video_key will be set by _upload_to_r2 after this)
        storage.save_reel_data(reel_id, {**reel_data, "reel_id": reel_id}, video_path=None)
        # Overlay video insights
        if hasattr(storage, "supabase"):
            update = {
                "video_insights":         insights,
                "video_processed":        True,
                "inferred_duration_days": insights.get("inferred_duration_days"),
                "inferred_budget_level":  insights.get("inferred_budget_level"),
            }
            storage.supabase.table("reel_cache").update(update).eq("reel_id", reel_id).execute()
        print(f"💾 Supabase updated for reel: {reel_id}")
    except Exception as e:
        print(f"⚠️  Supabase save failed: {e}")


def _upload_to_r2(reel_id: str, video_path: Optional[str]) -> None:
    """Upload downloaded video to Cloudflare R2 and write r2_video_key back to Supabase."""
    if not video_path or not storage or not hasattr(storage, "s3"):
        return
    try:
        key = f"{reel_id}.mp4"
        with open(video_path, "rb") as f:
            storage.s3.upload_fileobj(
                f, storage.r2_bucket, key,
                ExtraArgs={"ContentType": "video/mp4", "CacheControl": "public, max-age=31536000"},
            )
        # Write r2_video_key so the next cache hit skips CDN download
        if hasattr(storage, "supabase"):
            storage.supabase.table("reel_cache") \
                .update({"r2_video_key": key}) \
                .eq("reel_id", reel_id).execute()
        print(f"☁️  Video uploaded to R2: {key}")
    except Exception as e:
        print(f"⚠️  R2 upload failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SSE STREAMING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

async def _analyze_reel_stream(reel_url: str, skip_audio: bool, skip_video: bool) -> AsyncGenerator[str, None]:
    """
    The core SSE generator.
    Yields formatted SSE event strings as each stage completes.
    The frontend listens to these and updates its progress UI in real time.
    """
    video_path  : Optional[str] = None
    audio_path  : Optional[str] = None
    frame_paths : List[str]     = []
    reel_id     : str           = extract_reel_id_from_url(reel_url)

    try:
        # ══════════════════════════════════════════════════════════════════
        # STEP 0 — Smart cache check
        # ══════════════════════════════════════════════════════════════════
        # Four possible states based on what is already in Supabase:
        #
        #   State A — Nothing cached
        #             → run full pipeline: Apify + video/audio
        #
        #   State B — Caption/hashtags cached, NO video insights
        #             → skip Apify, run video/audio pipeline only
        #
        #   State C — Video insights cached, NO caption/hashtags
        #             (unusual but possible if Apify failed first time)
        #             → run Apify, skip video/audio pipeline
        #
        #   State D — Both caption AND video insights cached  ← FULL HIT
        #             → skip everything, go straight to place extraction
        #
        # ══════════════════════════════════════════════════════════════════
        yield _sse("cache_check", "running", "Checking database for cached data…", 4)

        cached        = None
        has_metadata  = False   # caption + hashtags present in DB
        has_video     = False   # video_processed=True and video_insights present

        if storage and storage.exists(reel_id):
            cached       = storage.get_metadata(reel_id)
            has_metadata = bool(
                cached and (cached.get("caption") or cached.get("hashtags"))
            )
            has_video    = bool(
                cached and cached.get("video_processed") and cached.get("video_insights")
            )

        # ── State D: Full cache hit ───────────────────────────────────────
        if has_metadata and has_video:
            yield _sse("cache_check", "done",
                       "✅ Full cache hit — caption + video both available ⚡", 10)

            for step, msg, pct in [
                ("apify",        "Caption & hashtags loaded from cache",  18),
                ("download",     "Video already processed — skipped",     26),
                ("frames",       "Frames already analysed — skipped",     36),
                ("audio_extract","Audio already transcribed — skipped",   46),
                ("vision",       "Vision data loaded from cache",         56),
                ("whisper",      "Transcript loaded from cache",          66),
                ("consolidate",  "Insights loaded from cache",            76),
                ("save",         "Already in database — skipped",         84),
            ]:
                yield _sse(step, "done", msg, pct)

            yield _sse("places", "running", "Building place list from cache…", 90)
            insights = cached.get("video_insights", {})
            places_result = await _run(
                extract_places_from_all_data,
                cached.get("caption", ""),
                cached.get("hashtags", []),
                cached.get("location", ""),
                insights,
            )
            yield _sse("places", "done",
                       f"{len(places_result['places'])} places ready", 97)
            yield _sse("complete", "done", "Ready!", 100,
                       destination      = places_result["destination"],
                       country          = places_result.get("country", ""),
                       places           = places_result["places"],
                       video_insights   = insights,
                       cached_reel_data = {
                           "url":            cached.get("url", reel_url),
                           "caption":        cached.get("caption", ""),
                           "hashtags":       cached.get("hashtags", []),
                           "location":       cached.get("location", ""),
                           "video_insights": insights,
                       })
            return

        # ── State B: Caption cached, no video → skip Apify ───────────────
        elif has_metadata and not has_video:
            yield _sse("cache_check", "done",
                       "Caption & hashtags cached — running video analysis only…", 8)
            reel_data   = cached
            need_apify  = False
            need_video  = True
            print(f"🗄️  Cache state B: metadata hit, video missing for {reel_id}")

        # ── State C: Video cached, no caption → skip video pipeline ───────
        elif has_video and not has_metadata:
            yield _sse("cache_check", "done",
                       "Video insights cached — fetching caption from Instagram only…", 8)
            reel_data   = cached    # has video_insights but no caption
            need_apify  = True
            need_video  = False
            print(f"🗄️  Cache state C: video hit, metadata missing for {reel_id}")

        # ── State A: Nothing cached → full pipeline ───────────────────────
        else:
            yield _sse("cache_check", "done",
                       "No cache — running full pipeline…", 8)
            reel_data   = None
            need_apify  = True
            need_video  = True
            print(f"🆕  Cache state A: no cache for {reel_id}")


        # ── STEP 1: Apify fetch ───────────────────────────────────────────
        # Skipped in State B (caption already cached) and State D (full hit).
        if need_apify:
            yield _sse("apify", "running", "Fetching caption & metadata from Instagram…", 12)
            try:
                fresh = await _run(_apify_fetch, reel_url)
                # Merge fresh Apify data into any existing cached row
                # (preserves cached video_insights if this is State C)
                if reel_data:
                    reel_data = {**reel_data, **fresh}
                else:
                    reel_data = fresh
                hashtag_count   = len(reel_data.get("hashtags", []))
                caption_preview = (reel_data.get("caption") or "")[:60]
                yield _sse("apify", "done",
                           f"{hashtag_count} hashtags · \"{caption_preview}…\"", 28)
            except Exception as e:
                yield _sse("apify", "failed", f"Apify error: {str(e)}", 28)
                yield _sse("error", "failed", str(e), 28)
                return
        else:
            yield _sse("apify", "done",
                       f"Caption cached — {len(reel_data.get('hashtags',[]))} hashtags ✓", 28)

        # ── STEP 2: Video download ────────────────────────────────────────
        # Skipped in State C (video insights already in cache) or skip_video flag.
        if not need_video or skip_video:
            reason = "Video insights already cached" if (not need_video) else "Video analysis skipped (caption-only mode)"
            yield _sse("download",     "skipped", reason, 35)
            yield _sse("frames",       "skipped", reason, 43)
            yield _sse("audio_extract","skipped", reason, 43)
            yield _sse("vision",       "skipped", reason, 55)
            yield _sse("whisper",      "skipped", reason, 55)
            yield _sse("consolidate",  "skipped", reason, 76)
        else:
            # Priority: R2 (already uploaded) → Instagram CDN → skip
            r2_key    = (reel_data or {}).get("r2_video_key", "")
            video_url = (reel_data or {}).get("video_url", "")

            if r2_key:
                yield _sse("download", "running", f"Loading video from R2 cache ({r2_key})…", 32)
                video_path = await _run(_download_video_from_r2, r2_key)
                if video_path:
                    size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    yield _sse("download", "done", f"Loaded from R2: {size_mb:.1f} MB ⚡", 40)
                else:
                    # R2 failed — fall back to CDN
                    yield _sse("download", "running", "R2 failed — downloading from Instagram CDN…", 35)
                    video_path = await _run(_download_video, video_url)
                    if video_path:
                        size_mb = os.path.getsize(video_path) / (1024 * 1024)
                        yield _sse("download", "done", f"CDN fallback: {size_mb:.1f} MB", 40)
                    else:
                        yield _sse("download", "skipped", "Download failed — using caption only", 40)

            elif video_url:
                yield _sse("download", "running", "Downloading reel video from Instagram…", 32)
                video_path = await _run(_download_video, video_url)
                if video_path:
                    size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    yield _sse("download", "done", f"Downloaded {size_mb:.1f} MB", 40)
                else:
                    yield _sse("download", "skipped", "Download failed — using caption only", 40)

            else:
                yield _sse("download", "skipped", "No video URL available — using caption only", 40)

        # ── STEPS 3-5: Frame/Audio/Vision/Whisper/Consolidation ─────────
        # These steps are skipped entirely in State C (need_video=False),
        # in which case we reuse the video_insights already in the DB row.

        if need_video and not skip_video:
            # ── STEP 3: Frame + Audio extraction — CONCURRENT ────────────
            if video_path:
                yield _sse("frames",       "running", "Extracting video frames (ffmpeg)…", 43)
                yield _sse("audio_extract","running", "Stripping audio track (ffmpeg)…",  43)

                frame_task = _run(extract_frames, video_path)
                audio_task = _run(extract_audio, video_path)
                frame_paths, audio_path = await asyncio.gather(frame_task, audio_task)

                if frame_paths:
                    yield _sse("frames", "done", f"{len(frame_paths)} frames extracted", 52)
                else:
                    yield _sse("frames", "skipped", "No frames extracted", 52)

                if audio_path:
                    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    yield _sse("audio_extract", "done", f"Audio ready ({size_mb:.1f} MB)", 52)
                else:
                    yield _sse("audio_extract", "skipped", "No audio stream found", 52)
            else:
                yield _sse("frames",       "skipped", "No video downloaded — skipped", 52)
                yield _sse("audio_extract","skipped", "No video downloaded — skipped", 52)

            # ── STEP 4: Vision + Whisper — CONCURRENT ────────────────────
            vision_data: dict = {}
            audio_data:  dict = {}

            vision_task  = None
            whisper_task = None

            if frame_paths:
                yield _sse("vision", "running", f"GPT-4o analysing {len(frame_paths)} frames…", 55)
                vision_task = asyncio.create_task(_run(analyse_frames_with_vision, frame_paths))

            if audio_path and not skip_audio:
                yield _sse("whisper", "running", "Whisper transcribing voiceover…", 55)
                whisper_task = asyncio.create_task(_run(transcribe_audio, audio_path))
            elif skip_audio:
                yield _sse("whisper", "skipped", "Audio transcription skipped", 55)
            else:
                yield _sse("whisper", "skipped", "No audio track", 55)

            if vision_task and whisper_task:
                vision_data, audio_data = await asyncio.gather(vision_task, whisper_task)
            elif vision_task:
                vision_data = await vision_task
            elif whisper_task:
                audio_data  = await whisper_task

            if vision_data:
                yield _sse("vision", "done",
                           f"{len(vision_data.get('places',[]))} places, "
                           f"{len(vision_data.get('scene_types',[]))} scenes detected", 68)
            else:
                yield _sse("vision", "skipped", "No frames to analyse", 68)

            if audio_data and audio_data.get("has_speech"):
                yield _sse("whisper", "done",
                           f"{audio_data.get('word_count',0)} words · "
                           f"{len(audio_data.get('itinerary_cues',[]))} cues", 72)
            elif audio_data:
                yield _sse("whisper", "done", "Audio has no speech (music only)", 72)

            # ── STEP 5: Consolidation ─────────────────────────────────────
            yield _sse("consolidate", "running", "Merging all signals…", 76)

            insights = await _run(
                consolidate_video_insights,
                vision_data, audio_data,
                reel_data.get("caption", ""),
                reel_data.get("hashtags", []),
            )

            dur    = insights.get("inferred_duration_days")
            budget = insights.get("inferred_budget_level", "")
            vibe   = insights.get("vibe", "")
            yield _sse("consolidate", "done",
                       f"{dur}-day {budget} {vibe} trip confirmed" if dur
                       else "All signals consolidated", 83)

        else:
            # State C: video was skipped — load insights directly from cache row
            insights = (reel_data or {}).get("video_insights") or {}
            n_places = len(insights.get("places", []))
            yield _sse("consolidate", "done",
                       f"Video insights loaded from cache ({n_places} places) ✓", 83)
            print(f"🗄️  Using cached video_insights for {reel_id}: {n_places} places")

        # ── STEP 6: Save to Supabase + Upload to R2 ─────────────────────
        # Only save what is genuinely new to avoid overwriting cached data.
        #   - need_apify=True  → we fetched fresh caption/hashtags, save them
        #   - need_video=True  → we ran video pipeline, save insights + upload
        #   - Both False       → nothing new to save (State D handled above)
        if need_apify or need_video:
            yield _sse("save", "running", "Saving new data to database…", 86)
            tasks = []
            if need_apify or need_video:
                # Always upsert the full reel_data row (safe: has reel_id as key)
                tasks.append(_run(_save_to_supabase, reel_id, reel_data, insights
                                  if need_video else (reel_data.get("video_insights") or {})))
            if need_video and video_path:
                tasks.append(_run(_upload_to_r2, reel_id, video_path))

            await asyncio.gather(*tasks)
            yield _sse("save", "done", "Database updated ✓", 90)
        else:
            yield _sse("save", "skipped", "Nothing new to save", 90)

        # ── STEP 7: Place extraction (uses all consolidated data) ─────────
        yield _sse("places", "running", "Building place selection list…", 93)

        places_result = await _run(
            extract_places_from_all_data,
            reel_data.get("caption", ""),
            reel_data.get("hashtags", []),
            reel_data.get("location", ""),
            insights,
        )

        n_places = len(places_result.get("places", []))
        dest     = places_result.get("destination", "")
        yield _sse("places", "done", f"{n_places} places found in {dest}", 97)

        # ── FINAL: Complete event with full payload ───────────────────────
        yield _sse(
            "complete", "done", "Analysis complete — select your places!", 100,
            destination  = places_result["destination"],
            country      = places_result.get("country", ""),
            places       = places_result["places"],
            video_insights = insights,
            cached_reel_data = {
                "url":           reel_data.get("url", reel_url),
                "caption":       reel_data.get("caption", ""),
                "hashtags":      reel_data.get("hashtags", []),
                "location":      reel_data.get("location", ""),
                "video_insights": insights,
            },
        )

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        traceback.print_exc()
        yield _sse("error", "failed", f"Pipeline error: {str(e)}", 0)

    finally:
        # Always clean up temp files
        cleanup_temp_files(video_path, audio_path, frame_paths)


# ═══════════════════════════════════════════════════════════════════════════
# ITINERARY GENERATION (unchanged logic, now accepts video_insights)
# ═══════════════════════════════════════════════════════════════════════════

def _build_video_context(insights: Optional[dict]) -> str:
    if not insights:
        return ""
    places    = ", ".join(insights.get("places", [])) or "none"
    days      = ", ".join(insights.get("day_labels", [])) or "not visible"
    acts      = ", ".join(insights.get("activities", [])) or "none"
    scenes    = ", ".join(insights.get("scene_types", [])) or "unknown"
    hotels    = ", ".join(insights.get("hotels", [])) or "none"
    restaurants = ", ".join(insights.get("restaurants", [])) or "none"
    prices    = ", ".join(insights.get("prices", [])) or "none"
    cues      = "; ".join(insights.get("itinerary_cues", [])[:8]) or "no speech"
    dur       = insights.get("inferred_duration_days", "unknown")
    budget    = insights.get("inferred_budget_level", "unknown")
    vibe      = insights.get("vibe", "unknown")
    summary   = insights.get("raw_summary", "")

    return f"""

═══════════════════════════════════════
VIDEO INTELLIGENCE (HIGH CONFIDENCE — extracted from actual reel frames + audio)
Use this data with HIGH PRIORITY over inferences from caption/hashtags alone.
═══════════════════════════════════════
Places detected on screen : {places}
Day structure visible      : {days}
Activities seen            : {acts}
Scene types                : {scenes}
Trip vibe                  : {vibe}
Hotels/stays visible       : {hotels}
Restaurants visible        : {restaurants}
Prices visible             : {prices}
Inferred duration          : {dur} days
Inferred budget            : {budget}
Audio/voiceover cues       : {cues}
Video summary              : {summary}
═══════════════════════════════════════
"""


def generate_reel_itinerary(reel_data: dict, duration_override: Optional[int],
                             budget_level: Optional[str], selected_places: Optional[List[str]]) -> dict:

    caption      = reel_data.get("caption", "")
    hashtags     = reel_data.get("hashtags", [])
    location     = reel_data.get("location", "")
    insights     = reel_data.get("video_insights")

    hashtag_text = " ".join(f"#{t}" for t in hashtags)
    full_text    = f"{caption}\n\nHashtags: {hashtag_text}"
    if location:
        full_text += f"\n\nLocation tag: {location}"

    places_block = ""
    if selected_places:
        places_block = f"""
SELECTED PLACES (USER CHOSE THESE — MUST INCLUDE ALL):
{', '.join(selected_places)}
- Build the entire itinerary around these places
- Distribute across days by proximity or theme
- Every selected place MUST appear in at least one day
- Add complementary nearby spots to fill each day
"""
    else:
        places_block = "- Infer key places from the reel content and video intelligence"

    PROMPT = f"""
You are a travel expert creating a detailed itinerary from an Instagram Reel.

INSTAGRAM REEL DATA:
{full_text}
{_build_video_context(insights)}

CONSTRAINTS:
{f'- Trip duration: {duration_override} days' if duration_override else '- Infer duration from content'}
{f'- Budget level: {budget_level}' if budget_level else '- Infer budget from content'}
{places_block}

OUTPUT: Return ONLY valid JSON — no markdown, no explanations:

{{
  "destination": "string",
  "country": "string",
  "duration": number,
  "budget_level": "budget | mid-range | luxury",
  "theme": ["themes"],
  "vibe": "string",
  "days": [
    {{
      "day": number,
      "title": "string",
      "activities": [
        {{
          "name": "string",
          "start_time": "9:00 AM",
          "end_time": "11:00 AM",
          "duration": "2 hours",
          "description": "string",
          "category": "sightseeing | food | adventure | relaxation | culture"
        }}
      ],
      "food": [
        {{
          "meal_type": "Breakfast | Lunch | Dinner | Snacks",
          "time": "string",
          "restaurant_name": "string",
          "location": "string",
          "dishes": [{{"name":"string","description":"string"}}],
          "price_range": "$ | $$ | $$$",
          "why_recommended": "string"
        }}
      ],
      "accommodation": {{
        "type": "hotel | hostel | resort | homestay | camping",
        "area": "string",
        "suggestion": "string"
      }}
    }}
  ],
  "key_highlights": ["string"],
  "estimated_budget": {{
    "total": "string",
    "breakdown": {{"accommodation":"string","food":"string","activities":"string","transport":"string"}}
  }},
  "travel_tips": ["string"],
  "best_time_to_visit": "string",
  "packing_suggestions": ["string"]
}}

Rules:
- Start each day 8–9 AM, end 9–10 PM
- Include Breakfast, Lunch, Dinner every day — use REAL dish names
- ALL activities must have start_time, end_time, duration
- Use video intelligence places as the PRIMARY source for activities
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Travel planning AI. Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": PROMPT},
            ],
            temperature=0.7,
        )
        content   = response.choices[0].message.content.strip()
        itinerary = _json(content)
        required  = {"destination", "duration", "budget_level", "days"}
        missing   = required - itinerary.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        print(f"✅ Itinerary: {itinerary['destination']} ({itinerary['duration']} days)")
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Itinerary generation failed: {str(e)}")


def generate_manual_itinerary(req: ManualItineraryRequest) -> dict:
    PROMPT = """
Act as an expert travel consultant. Create a highly detailed, optimised itinerary.

DESTINATION: {place}
DURATION: {duration} days
THEME: {theme}
TRAVELERS: {travelers}
BUDGET: {budget}
INTERESTS: {interests}
PACE: {pace}
ACCOMMODATION: {accommodation}
TRANSPORT: {transport}
FOOD: {food}
CONSTRAINTS: {constraints}
SEASON: {season}

Return ONLY valid JSON starting with {{ ending with }}.
No markdown. No explanations.

Required top-level keys: destination, duration, budget_level, days, travel_tips

Each day must have activities (with start_time, end_time, duration) and food (Breakfast, Lunch, Dinner with REAL dish names).
Food schema: meal_type, time, restaurant_name, location, dishes[{{name,description}}], price_range ($/$$/$$$), why_recommended.
Activity schema: name, start_time, end_time, duration, description.
""".format(
        place=req.place, duration=req.duration, theme=", ".join(req.theme),
        travelers=req.number_of_people, budget=req.budget_level,
        interests=", ".join(req.interests), pace=req.pace,
        accommodation=req.accommodation_area, transport=req.transport_preference,
        food=req.food_preference,
        constraints=", ".join(req.constraints) if req.constraints else "None",
        season=", ".join(req.season_dates) if req.season_dates else "Not specified",
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Strict JSON API. Travel itineraries. No markdown."},
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.5,
    )
    content = response.choices[0].message.content.strip()
    itinerary = _json(content)

    required = {"destination", "duration", "budget_level", "days", "travel_tips"}
    missing  = required - itinerary.keys()
    if missing:
        raise RuntimeError(f"Missing required keys: {missing}")

    for day in itinerary.get("days", []):
        acts = []; food = []
        for item in day.get("activities", []):
            (food if "meal_type" in item else acts).append(item)
        day["activities"] = acts
        day["food"]       = food

    return itinerary


# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "app": "TrailBuddy Travel Itinerary Generator",
        "version": "4.0",
        "endpoints": {
            "POST /analyze-reel":         "SSE stream — full concurrent reel analysis pipeline",
            "POST /generate-from-reel":   "Generate itinerary from analyzed reel data",
            "POST /generate":             "Generate itinerary from manual input",
            "GET  /health":               "Health check",
        },
    }


@app.get("/health")
async def health():
    return {
        "status":            "healthy",
        "openai":            bool(OPENAI_API_KEY),
        "apify":             bool(APIFY_API_TOKEN),
        "storage":           storage.__class__.__name__ if storage else "disabled",
        "ffmpeg":            ffmpeg_available(),
        "storage_backend":   STORAGE_BACKEND,
    }


@app.post("/analyze-reel")
async def analyze_reel(request: AnalyzeReelRequest):
    """
    *** PRIMARY ENDPOINT for Instagram reel analysis ***

    Streams real-time progress events (Server-Sent Events) for every stage
    of the concurrent processing pipeline.

    Event format:
        data: {"step":"apify","status":"running","message":"...","progress":15}

    Steps (in order):
        cache_check → apify → download → frames + audio_extract (concurrent)
        → vision + whisper (concurrent) → consolidate → save → places → complete

    Final "complete" event carries:
        destination, country, places[], video_insights{}, cached_reel_data{}

    The frontend stores cached_reel_data and passes it back to /generate-from-reel.
    """
    return StreamingResponse(
        _analyze_reel_stream(request.reel_url, request.skip_audio, request.skip_video),
        media_type="text/event-stream",
        headers={
            "Cache-Control":            "no-cache",
            "X-Accel-Buffering":        "no",   # disable Nginx buffering
            "Connection":               "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/generate-from-reel")
async def generate_from_reel(request: GenerateFromReelRequest):
    """
    Generate a full day-by-day itinerary using data from /analyze-reel.

    The frontend passes back:
      - cached_reel_data  (from the "complete" event of /analyze-reel)
      - video_insights    (same complete event)
      - selected_places   (the user's place selection)
      - duration_override / budget_level (optional overrides)
    """
    try:
        reel_data = request.cached_reel_data or {}
        if not reel_data:
            raise HTTPException(status_code=400, detail="cached_reel_data is required")

        # Merge video_insights into reel_data
        if request.video_insights:
            reel_data["video_insights"] = request.video_insights
        elif not reel_data.get("video_insights") and storage and hasattr(storage, "supabase"):
            # Try to load from Supabase as a fallback
            reel_id = extract_reel_id_from_url(request.reel_url)
            try:
                result = storage.supabase.table("reel_cache").select(
                    "video_insights").eq("reel_id", reel_id).execute()
                if result.data:
                    reel_data["video_insights"] = result.data[0].get("video_insights")
            except Exception:
                pass

        itinerary = generate_reel_itinerary(
            reel_data,
            duration_override=request.duration_override,
            budget_level=request.budget_level,
            selected_places=request.selected_places,
        )

        return {
            "success": True,
            "source": "instagram",
            "video_enhanced": bool(reel_data.get("video_insights")),
            "data": itinerary,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_manual(request: ManualItineraryRequest):
    """Generate itinerary from manual form input."""
    try:
        itinerary = generate_manual_itinerary(request)
        return {"success": True, "data": itinerary, "source": "manual"}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Backward-compatibility shims ─────────────────────────────────────────────
# These endpoints existed in the old two-step flow.
# Old frontend versions (or cached HTML) may still call them.
# They are re-implemented here so deployments with stale frontends don't 404.

class _ExtractPlacesRequest(BaseModel):
    reel_url: str

@app.post("/extract-places-from-reel")
async def compat_extract_places(request: _ExtractPlacesRequest):
    """
    Legacy endpoint (old two-step flow, Step 1).
    Kept for backward compatibility with deployed frontends that haven't
    received the new index.html yet.

    Calls the same Apify + LLM logic as before and returns the same shape
    the old frontend expected, so nothing breaks during the transition.
    """
    try:
        reel_data     = await _run(_apify_fetch, request.reel_url)
        reel_id       = extract_reel_id_from_url(request.reel_url)
        places_result = await _run(
            extract_places_from_all_data,
            reel_data.get("caption", ""),
            reel_data.get("hashtags", []),
            reel_data.get("location", ""),
            {},   # no video insights in legacy flow
        )
        return {
            "success": True,
            "reel_data": {
                "url":      reel_data["url"],
                "caption":  (reel_data.get("caption") or "")[:300],
                "hashtags": reel_data.get("hashtags", []),
                "location": reel_data.get("location", ""),
            },
            "cached_reel_data": {
                "url":      reel_data["url"],
                "caption":  reel_data.get("caption", ""),
                "hashtags": reel_data.get("hashtags", []),
                "location": reel_data.get("location", ""),
                "likes":    reel_data.get("likes", 0),
            },
            "destination": places_result.get("destination", ""),
            "country":     places_result.get("country", ""),
            "places":      places_result.get("places", []),
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)