"""
video_processor.py — TrailBuddy Video Intelligence Pipeline
============================================================
Pure synchronous functions for each stage of the video pipeline.
Designed to be called from asyncio.run_in_executor() so multiple
stages can run truly concurrently inside the SSE streaming handler.

Pipeline stages:
  1. extract_frames(video_path)           → List[str]  (JPEG paths)
  2. extract_audio(video_path)            → str | None (MP3 path)
  3. analyse_frames_with_vision(frames)   → dict       (OCR + scene data)
  4. transcribe_audio(audio_path)         → dict       (transcript + cues)
  5. consolidate_video_insights(...)      → dict       (VideoInsights)
  6. extract_places_from_all_data(...)    → dict       (place list for UI)
  7. cleanup_temp_files(...)              → None
"""

import os
import json
import base64
import tempfile
import subprocess
import traceback
from typing import Optional, List, Dict, Any
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
# Frame extraction strategy
DENSE_INTERVAL_SEC  = 0.5    # sample every 0.5s = 2 fps, catches fast text overlays
SCENE_THRESHOLD     = 0.25   # ffmpeg scene-change sensitivity (0=always, 1=never)
                              # 0.25 captures any meaningful visual change
MAX_UNIQUE_FRAMES   = 60     # hard cap on unique frames kept after dedup
OCR_BATCH_SIZE      = 15     # frames per GPT-4o Vision call (API limit ~20)

# Audio
MAX_AUDIO_MB        = 24     # Whisper hard limit = 25 MB

# Models
VISION_MODEL        = "gpt-4o"
WHISPER_MODEL       = "whisper-1"

# ── OpenAI client (lazy) ─────────────────────────────────────────────────────
_openai_client: Optional[OpenAI] = None

def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=key)
    return _openai_client

def ffmpeg_available() -> bool:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


# ===========================================================================
# STAGE 1 — Frame Extraction  (scene-change-aware, dense sampling)
# ===========================================================================
#
# Strategy:
#   Pass A — scene-change frames (ffmpeg select filter, threshold 0.25):
#       ffmpeg outputs a frame only when the image differs significantly from
#       the previous one. This catches every text slide, every cut, every new
#       location card — even sub-second overlays that a fixed 2s interval misses.
#
#   Pass B — dense uniform frames (every 0.5s):
#       Fills gaps left by the scene filter (e.g. slow-zoom shots with text that
#       never crosses the change threshold). Guarantees no frame is missed.
#
#   Deduplication — pixel-diff hash (no PIL needed, pure ffmpeg):
#       After combining both passes, consecutive near-identical frames are
#       dropped using ffmpeg's "freezedetect" output. This prevents sending
#       50 frames of the same static slide to the Vision API.
#
#   Result: up to MAX_UNIQUE_FRAMES unique visually-distinct frames, sorted
#   by timestamp, ready for OCR.

def extract_frames(video_path: str) -> List[str]:
    """
    Extract visually unique frames from a travel reel using two complementary
    ffmpeg passes:
      1. Scene-change detection  — catches every cut, slide change, text overlay
      2. Dense uniform sampling  — catches slow-motion / slow-zoom shots

    The two sets are merged, sorted by timestamp, and deduplicated by
    comparing adjacent frames with ffmpeg's signalstats filter so no
    extra Python dependencies are needed.

    Returns list of absolute JPEG paths, capped at MAX_UNIQUE_FRAMES.
    """
    if not ffmpeg_available():
        raise EnvironmentError(
            "ffmpeg not found.\n"
            "  Ubuntu/Debian : sudo apt-get install -y ffmpeg\n"
            "  macOS         : brew install ffmpeg\n"
            "  Render.com    : add  apt = ['ffmpeg']  in render.yaml"
        )

    duration = _video_duration(video_path)
    print(f"ℹ️  Video duration: {duration:.1f}s")

    out_dir = tempfile.mkdtemp(prefix="tb_frames_")

    # ── Pass A: scene-change frames ────────────────────────────────────────
    scene_dir = os.path.join(out_dir, "scene")
    os.makedirs(scene_dir)
    _run_ffmpeg_extract(
        video_path,
        scene_dir,
        vf_filter=f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        fps_override=None,
        label="scene",
    )

    # ── Pass B: dense uniform frames (every DENSE_INTERVAL_SEC) ───────────
    dense_dir = os.path.join(out_dir, "dense")
    os.makedirs(dense_dir)
    _run_ffmpeg_extract(
        video_path,
        dense_dir,
        vf_filter=None,
        fps_override=DENSE_INTERVAL_SEC,
        label="dense",
    )

    # ── Merge both sets, sort by embedded timestamp in filename ───────────
    all_frames = (
        sorted(Path(scene_dir).glob("*.jpg")) +
        sorted(Path(dense_dir).glob("*.jpg"))
    )

    if not all_frames:
        print("⚠️  No frames extracted — falling back to basic 1fps extraction")
        return _extract_frames_basic(video_path, out_dir, duration)

    # ── Deduplicate: drop consecutive near-identical frames ───────────────
    unique_paths = _deduplicate_frames([str(p) for p in all_frames])

    # ── Sort chronologically (dense frames are named with padded index) ───
    # Re-sort by file mtime is unreliable; use ffprobe per-frame pts instead.
    # For simplicity, sort dense first (uniform coverage) then scene frames
    # interleaved — the LLM reads the batch holistically so order matters less
    # than coverage. Final sort: alphabetical within each subdir keeps temporal order.
    unique_paths = sorted(unique_paths, key=lambda p: (
        0 if "dense" in p else 1,
        p
    ))

    # Interleave: take frames alternately from dense and scene so the Vision
    # API sees a balanced temporal spread rather than all dense then all scene.
    dense_frames = [p for p in unique_paths if "dense" in p]
    scene_frames = [p for p in unique_paths if "scene" in p]
    interleaved  = []
    d, s = 0, 0
    while d < len(dense_frames) or s < len(scene_frames):
        if d < len(dense_frames):
            interleaved.append(dense_frames[d]); d += 1
        if s < len(scene_frames):
            interleaved.append(scene_frames[s]); s += 1

    # Remove exact duplicates that appear in both sets
    seen_sizes: set = set()
    final: List[str] = []
    for p in interleaved:
        try:
            size = os.path.getsize(p)
            if size not in seen_sizes:
                seen_sizes.add(size)
                final.append(p)
        except OSError:
            pass

    # Enforce hard cap
    if len(final) > MAX_UNIQUE_FRAMES:
        step  = len(final) / MAX_UNIQUE_FRAMES
        final = [final[int(i * step)] for i in range(MAX_UNIQUE_FRAMES)]

    print(
        f"🎞️  {len(dense_frames)} dense + {len(scene_frames)} scene-change frames "
        f"→ {len(final)} unique kept (cap={MAX_UNIQUE_FRAMES})"
    )
    return final


def _run_ffmpeg_extract(video_path: str, out_dir: str,
                         vf_filter: Optional[str],
                         fps_override: Optional[float],
                         label: str) -> None:
    """Run a single ffmpeg frame-extraction pass into out_dir."""
    pattern = os.path.join(out_dir, f"frame_%05d.jpg")

    vf_parts = []
    if fps_override is not None:
        vf_parts.append(f"fps=1/{fps_override}")
    if vf_filter:
        vf_parts.append(vf_filter)

    cmd = ["ffmpeg", "-i", video_path]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += [
        "-q:v", "1",           # maximum JPEG quality (1 = ~98%)
        "-vsync", "vfr",       # variable frame rate output
        "-hide_banner", "-loglevel", "error",
        pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    n = len(list(Path(out_dir).glob("*.jpg")))
    if result.returncode != 0 and n == 0:
        print(f"  ⚠️  ffmpeg {label} pass warning: {result.stderr[:200]}")
    else:
        print(f"  ✅ {label} pass: {n} frames")


def _deduplicate_frames(paths: List[str]) -> List[str]:
    """
    Remove near-identical consecutive frames using file-size comparison as a
    fast proxy for visual similarity. Two frames with sizes within 3% of each
    other are almost certainly the same static slide.

    Falls back to keeping all frames if the list is small.
    """
    if len(paths) <= 5:
        return paths

    kept: List[str] = [paths[0]]
    for path in paths[1:]:
        try:
            prev_size = os.path.getsize(kept[-1])
            curr_size = os.path.getsize(path)
            if prev_size == 0:
                kept.append(path)
                continue
            diff_pct = abs(curr_size - prev_size) / prev_size
            if diff_pct > 0.03:   # more than 3% size difference = new content
                kept.append(path)
        except OSError:
            kept.append(path)

    removed = len(paths) - len(kept)
    if removed:
        print(f"  🗑️  Removed {removed} near-duplicate frames")
    return kept


def _extract_frames_basic(video_path: str, out_dir: str, duration: float) -> List[str]:
    """
    Emergency fallback: simple 1-fps extraction if both scene + dense passes fail.
    """
    pattern = os.path.join(out_dir, "fallback_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "1",
        "-hide_banner", "-loglevel", "error",
        pattern,
    ], capture_output=True)
    frames = sorted(Path(out_dir).glob("fallback_*.jpg"))
    return [str(f) for f in frames[:MAX_UNIQUE_FRAMES]]


def _video_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

# ===========================================================================
# STAGE 2 — Audio Extraction (ffmpeg)
# ===========================================================================

def extract_audio(video_path: str) -> Optional[str]:
    """
    Strip audio track from video into a temp MP3.
    Returns MP3 path or None if no audio / ffmpeg unavailable.
    """
    if not ffmpeg_available():
        print("⚠️  ffmpeg unavailable — skipping audio")
        return None

    out_path = video_path.rsplit(".", 1)[0] + "_audio.mp3"

    result = subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        "-hide_banner", "-loglevel", "error",
        out_path,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ℹ️  No audio stream found (code {result.returncode})")
        return None

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"🔊 Audio extracted: {size_mb:.2f} MB → {out_path}")

    if size_mb > MAX_AUDIO_MB:
        print(f"⚠️  Audio {size_mb:.1f} MB > limit — trimming")
        out_path = _trim_audio(out_path)

    return out_path


def _trim_audio(path: str) -> str:
    duration = _audio_duration(path)
    size_mb  = os.path.getsize(path) / (1024 * 1024)
    safe_sec = int((MAX_AUDIO_MB / size_mb) * duration * 0.9)
    trimmed  = path.replace(".mp3", "_trimmed.mp3")
    subprocess.run([
        "ffmpeg", "-i", path, "-t", str(safe_sec), "-c", "copy",
        "-hide_banner", "-loglevel", "error", trimmed,
    ], capture_output=True)
    return trimmed


def _audio_duration(path: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 120.0


# ===========================================================================
# STAGE 3 — GPT-4o Vision (Batched Parallel OCR + Scene Analysis)
# ===========================================================================
#
# All frames are sent at detail="high" — no more low-detail scene pass.
# Reason: travel reels pack every useful signal into text overlays; the
# scene-type vibe is a secondary concern and GPT-4o reads it fine at
# high detail anyway.
#
# Batching strategy:
#   Frames are split into batches of OCR_BATCH_SIZE (15).
#   Each batch is sent as a separate API call so the model can give
#   full attention to each set of frames without context overflow.
#   All batch API calls run CONCURRENTLY using ThreadPoolExecutor.
#   Results are merged with deduplication.
#
# Per-batch max_tokens = 2000 (each frame at high detail ≈ 765 tokens,
# so 15 frames ≈ 11,475 input tokens; 2000 output is plenty for JSON).

import concurrent.futures as _cf


def analyse_frames_with_vision(frame_paths: List[str]) -> Dict[str, Any]:
    """
    High-accuracy batched OCR + scene analysis.

    Every frame is sent at detail="high".
    Frames are processed in parallel batches of OCR_BATCH_SIZE.
    All batch results are merged with deduplication.
    """
    if not frame_paths:
        return _empty_vision_result()

    client = _openai()

    # ── Split into batches ─────────────────────────────────────────────────
    batches = [
        frame_paths[i : i + OCR_BATCH_SIZE]
        for i in range(0, len(frame_paths), OCR_BATCH_SIZE)
    ]
    print(
        f"🔭 Vision: {len(frame_paths)} frames → "
        f"{len(batches)} batch(es) of ≤{OCR_BATCH_SIZE}, all at detail=high"
    )

    # ── Run all batches concurrently ───────────────────────────────────────
    batch_results: List[Dict[str, Any]] = []

    with _cf.ThreadPoolExecutor(max_workers=min(len(batches), 6)) as pool:
        futures = {
            pool.submit(_vision_batch, client, batch, idx): idx
            for idx, batch in enumerate(batches)
        }
        for future in _cf.as_completed(futures):
            idx    = futures[future]
            result = future.result()
            batch_results.append(result)
            n_places = len(result.get("places", []))
            n_ocr    = len(result.get("ocr_text", []))
            n_days   = len(result.get("day_labels", []))
            print(f"  ✅ Batch {idx+1}/{len(batches)}: "
                  f"{n_ocr} OCR strings, {n_places} places, {n_days} day labels")

    # ── Merge all batch results ────────────────────────────────────────────
    merged = _merge_batch_results(batch_results)

    print(
        f"✅ Vision total: {len(merged.get('ocr_text',[]))} OCR strings, "
        f"{len(merged.get('places',[]))} places, "
        f"{len(merged.get('day_labels',[]))} day labels, "
        f"{len(merged.get('scene_types',[]))} scenes"
    )
    return merged


def _vision_batch(client: OpenAI, frame_paths: List[str], batch_idx: int) -> Dict[str, Any]:
    """
    Send one batch of frames to GPT-4o Vision at detail="high".
    Returns structured dict with OCR text, places, day labels, etc.
    """
    image_blocks = _build_image_blocks(frame_paths, detail="high")

    prompt = f"""You are analysing batch {batch_idx + 1} of frames from a travel Instagram reel.
These are real video frames — they may contain text overlays, subtitle cards, day labels,
price tags, itinerary slides, hotel names, place names, and scene visuals.

Extract EVERYTHING with maximum accuracy.

Return ONLY a valid JSON object — no markdown, no explanation:
{{
  "ocr_text": [
    "list every exact text string visible across ALL frames in this batch",
    "include subtitles, overlays, caption cards, itinerary slides",
    "include text in any language — transliterate Hindi/Tamil/etc to English",
    "if text is partially cut off, include what you can see with (?) suffix"
  ],
  "day_labels": [
    "Day 1", "Day 2", "etc — ONLY if you can actually READ them as text"
  ],
  "places": [
    "place names you can READ from text overlays OR confidently identify from imagery"
  ],
  "hotels": ["hotel, hostel, resort, or stay names visible as text"],
  "restaurants": ["restaurant, cafe, dhaba, food stall names visible as text"],
  "prices": ["any price strings: ₹1200, $50, Rs.800/night, INR 500, etc."],
  "duration_cues": ["trip length from text: '3 day trip', '5 days', 'weekend', etc."],
  "budget_signals": ["text or visual cues about budget: 'budget hotel', 'luxury resort', etc."],
  "activities": [
    "activities visible in text overlays OR happening in the frame visually",
    "e.g. rafting, trekking, paragliding, scuba, cafe hopping, temple visit"
  ],
  "scene_types": [
    "visual scene categories: beach, mountain, temple, waterfall, city, jungle,",
    "market, hotel room, restaurant, fort, lake, river, street, etc."
  ],
  "vibe": "adventure | luxury | backpacking | romantic | family | food-tour | cultural | offbeat | mixed",
  "itinerary_slides": [
    "full text of any itinerary slide, day plan, or schedule card visible in any frame"
  ],
  "raw_summary": "1-2 sentences about what this batch of frames shows"
}}

Critical rules:
- Return ONLY what you can actually see — do NOT hallucinate
- Blurry text: include your best read with (?) at end
- Regional scripts: transliterate (गोवा → Goa, दिल्ली → Delhi)
- day_labels: only real day markers you can read, not inferred
- itinerary_slides: copy the FULL visible text of any slide showing a plan or schedule
"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a high-accuracy travel data extraction AI. "
                        "Read every visible text string. Return ONLY valid JSON. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *image_blocks],
                },
            ],
            max_tokens=2000,
            temperature=0.0,   # deterministic for OCR accuracy
        )
        raw    = _strip_fences(response.choices[0].message.content.strip())
        result = json.loads(raw)
        # Ensure all expected keys exist
        for key in ["ocr_text", "day_labels", "places", "hotels", "restaurants",
                    "prices", "duration_cues", "budget_signals", "activities",
                    "scene_types", "itinerary_slides"]:
            result.setdefault(key, [])
        result.setdefault("vibe", "")
        result.setdefault("raw_summary", "")
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Batch {batch_idx+1} JSON parse error: {e}")
        return _empty_vision_result()
    except Exception as e:
        print(f"  ⚠️  Batch {batch_idx+1} failed: {e}")
        return _empty_vision_result()


def _merge_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from all vision batches.
    List fields: union with deduplication preserving first-seen order.
    String fields (vibe, raw_summary): take longest non-empty value.
    """
    list_keys = [
        "ocr_text", "day_labels", "places", "hotels", "restaurants",
        "prices", "duration_cues", "budget_signals", "activities",
        "scene_types", "itinerary_slides",
    ]
    merged: Dict[str, Any] = {k: [] for k in list_keys}
    seen:   Dict[str, set]  = {k: set() for k in list_keys}

    for result in results:
        for key in list_keys:
            for item in result.get(key, []):
                normalized = str(item).strip().lower()
                if normalized and normalized not in seen[key]:
                    seen[key].add(normalized)
                    merged[key].append(item)

    # Vibe: take the first non-empty value
    merged["vibe"] = next(
        (r.get("vibe", "") for r in results if r.get("vibe")), ""
    )
    # raw_summary: concatenate unique summaries
    summaries = [r.get("raw_summary", "") for r in results if r.get("raw_summary")]
    merged["raw_summary"] = " ".join(summaries[:3])   # first 3 batch summaries

    # Pull itinerary_slides into ocr_text as well (so consolidation sees them)
    for slide in merged.get("itinerary_slides", []):
        normalized = slide.strip().lower()
        if normalized and normalized not in seen["ocr_text"]:
            seen["ocr_text"].add(normalized)
            merged["ocr_text"].append(slide)

    return merged


def _build_image_blocks(frame_paths: List[str], detail: str) -> list:
    """Encode frames as base64 image_url blocks for the Vision API."""
    blocks = []
    for path in frame_paths:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
            })
        except Exception as e:
            print(f"  ⚠️  Could not encode frame {path}: {e}")
    return blocks


def _empty_vision_result() -> Dict[str, Any]:
    return {
        "ocr_text": [], "day_labels": [], "places": [], "hotels": [],
        "restaurants": [], "prices": [], "duration_cues": [],
        "budget_signals": [], "activities": [], "scene_types": [],
        "itinerary_slides": [], "vibe": "", "raw_summary": "",
    }


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        else:
            text = parts[-1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

# ===========================================================================
# STAGE 4 — Whisper Transcription
# ===========================================================================

def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe reel audio with Whisper. Returns transcript + structured cues.
    """
    client = _openai()
    print(f"🎙️  Transcribing audio...")

    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f,
                response_format="verbose_json",
                prompt="Instagram travel reel. Places, activities, hotels, food, durations, prices, tips.",
            )

        transcript = resp.text.strip()
        language   = getattr(resp, "language", "unknown")
        word_count = len(transcript.split())
        has_speech = word_count > 10

        print(f"✅ Whisper: {word_count} words, lang={language}, speech={has_speech}")

        cues = _extract_cues_from_transcript(transcript) if has_speech and len(transcript) > 30 else []

        return {
            "transcript": transcript, "detected_language": language,
            "has_speech": has_speech, "word_count": word_count, "itinerary_cues": cues,
        }

    except Exception as e:
        print(f"⚠️  Whisper failed: {e}")
        return {"transcript":"","detected_language":"unknown","has_speech":False,"word_count":0,"itinerary_cues":[]}


def _extract_cues_from_transcript(transcript: str) -> List[str]:
    client = _openai()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY a JSON array of strings. No markdown."},
                {"role": "user", "content": f"""
Extract travel facts from this transcript as short strings:
places, activities, food tips, stays, durations, costs, practical tips.

Transcript: {transcript[:3000]}

Return ONLY: ["fact 1", "fact 2", ...]
"""},
            ],
            max_tokens=600, temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ===========================================================================
# STAGE 5 — Consolidation
# ===========================================================================

def consolidate_video_insights(vision_data: Dict, audio_data: Dict,
                                caption: str = "", hashtags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Merge vision + audio + caption + hashtags → clean VideoInsights dict.
    Uses GPT-4o for deduplication and inference; falls back to manual merge on error.
    """
    client = _openai()
    transcript  = audio_data.get("transcript", "")

    prompt = f"""
You are merging travel signals from multiple sources into one structured object.

IMPORTANT: For the "places" field, keep EVERY distinct place name — do NOT merge or drop
places that seem similar. "Vagator Beach", "North Goa", and "Vagator" are all separate entries.
Fix spelling errors but preserve every individual place.

CAPTION: {caption[:800] or '(none)'}
HASHTAGS: {json.dumps((hashtags or []))}
OCR TEXT: {json.dumps(vision_data.get('ocr_text', []))}
ITINERARY SLIDES (full text of day-plan cards visible in video):
{json.dumps(vision_data.get('itinerary_slides', []))}
VIDEO PLACES: {json.dumps(vision_data.get('places', []))}
DAY LABELS: {json.dumps(vision_data.get('day_labels', []))}
SCENE TYPES: {json.dumps(vision_data.get('scene_types', []))}
ACTIVITIES: {json.dumps(vision_data.get('activities', []))}
HOTELS: {json.dumps(vision_data.get('hotels', []))}
RESTAURANTS: {json.dumps(vision_data.get('restaurants', []))}
PRICES: {json.dumps(vision_data.get('prices', []))}
DURATION CUES: {json.dumps(vision_data.get('duration_cues', []))}
BUDGET SIGNALS: {json.dumps(vision_data.get('budget_signals', []))}
AUDIO (first 1500 chars): {transcript[:1500] or '(no speech)'}
AUDIO CUES: {json.dumps(audio_data.get('itinerary_cues', []))}

Output schema (ONLY this JSON, no markdown):
{{
  "places": ["keep ALL distinct place names — fix spelling only, do not merge or drop any"],
  "day_labels": [],
  "activities": [],
  "scene_types": [],
  "vibe": "",
  "hotels": [],
  "restaurants": [],
  "prices": [],
  "duration_cues": [],
  "inferred_duration_days": null,
  "inferred_budget_level": "budget | mid-range | luxury",
  "budget_signals": [],
  "transcript": "",
  "itinerary_cues": [],
  "ocr_highlights": [],
  "raw_summary": ""
}}

Set inferred_duration_days to an integer only if you are confident. Otherwise null.
transcript = full audio transcript string.
ocr_highlights = top 10 most travel-useful OCR lines.
"""

    # Inject transcript safely
    prompt = prompt.replace('"transcript": ""', f'"transcript": {json.dumps(transcript)}')

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Consolidate travel data. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000, temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        result["video_processed"] = True
        result.setdefault("processing_notes", [])
        print(f"✅ Consolidation: {len(result.get('places',[]))} places, "
              f"{result.get('inferred_duration_days','?')}d, vibe={result.get('vibe','?')}")
        return result
    except Exception as e:
        print(f"⚠️  Consolidation LLM failed: {e} — manual merge")
        return {
            "places": list(set(vision_data.get("places", []))),
            "day_labels": vision_data.get("day_labels", []),
            "activities": list(set(vision_data.get("activities", []))),
            "scene_types": list(set(vision_data.get("scene_types", []))),
            "vibe": vision_data.get("vibe", ""),
            "hotels": vision_data.get("hotels", []),
            "restaurants": vision_data.get("restaurants", []),
            "prices": vision_data.get("prices", []),
            "duration_cues": vision_data.get("duration_cues", []),
            "inferred_duration_days": None,
            "inferred_budget_level": None,
            "budget_signals": vision_data.get("budget_signals", []),
            "transcript": audio_data.get("transcript", ""),
            "itinerary_cues": audio_data.get("itinerary_cues", []),
            "ocr_highlights": vision_data.get("ocr_text", [])[:10],
            "raw_summary": vision_data.get("raw_summary", ""),
            "video_processed": True,
            "processing_notes": ["LLM consolidation failed — manual merge used"],
        }


# ===========================================================================
# STAGE 6 — Place Extraction (uses ALL data for richer results)
# ===========================================================================

def _segment_hashtag(tag: str) -> str:
    """
    Split a run-together hashtag into readable words using simple heuristics.
    e.g. "vagatorbeach" → "Vagator Beach"
         "chapofort"    → "Chapo Fort"   (LLM will fix spelling later)
         "northgoatrip" → "North Goa Trip"
    Uses a lightweight word-boundary detection based on a built-in word list.
    Falls back to the raw tag if segmentation is ambiguous.
    """
    import re
    # Common travel suffixes/prefixes that are word boundaries
    boundary_words = [
        "beach", "fort", "lake", "river", "falls", "temple", "market", "cafe",
        "hotel", "resort", "island", "village", "city", "town", "north", "south",
        "east", "west", "trip", "tour", "travel", "goa", "bali", "kerala",
        "mumbai", "delhi", "jaipur", "udaipur", "manali", "shimla", "ladakh",
        "rishikesh", "mussoorie", "ooty", "coorg", "munnar", "alleppey",
        "pondicherry", "varanasi", "agra", "jaisalmer", "pushkar", "hampi",
    ]
    tag_lower = tag.lower()
    # Try to find known word boundaries
    result = tag_lower
    for word in sorted(boundary_words, key=len, reverse=True):  # longest first
        result = result.replace(word, f" {word} ")
    # Collapse spaces, title-case
    clean = " ".join(result.split()).title()
    return clean if len(clean) > 2 else tag.title()


def extract_places_from_all_data(caption: str, hashtags: List[str],
                                  location: str, video_insights: Dict) -> Dict:
    """
    Build the user-facing place selection list using ALL sources.

    Key fixes vs previous version:
    - Hashtags sent as a full JSON list (not a truncated joined string)
    - Each hashtag is pre-segmented so the LLM reads "Vagator Beach" not "vagatorbeach"
    - Prompt instructs the LLM to KEEP all distinct places, NOT merge similar ones
    - Runs two LLM passes: one for hashtag/caption, one for video — then combines
      This mirrors the old approach's thoroughness while adding video signals
    """
    client = _openai()

    # ── Pre-process hashtags into readable form ──────────────────────────
    # Send the full list — no truncation. Average hashtag is ~12 chars,
    # so even 50 hashtags = ~600 chars which is fine in a prompt.
    segmented_hashtags = [_segment_hashtag(h) for h in hashtags]

    # ── Build video signal strings ────────────────────────────────────────
    video_places     = video_insights.get("places", [])
    video_hotels     = video_insights.get("hotels", [])
    video_food       = video_insights.get("restaurants", [])
    video_acts       = video_insights.get("activities", [])
    audio_cues       = video_insights.get("itinerary_cues", [])
    ocr_text         = video_insights.get("ocr_highlights", video_insights.get("ocr_text", []))[:15]
    itinerary_slides = video_insights.get("itinerary_slides", [])  # full text of plan slides

    prompt = f"""
You are extracting a COMPLETE list of places from an Instagram travel reel.
Your goal is to find EVERY individual place — err on the side of MORE places, not fewer.

=== SOURCE DATA ===

CAPTION:
{caption[:1000] or '(none)'}

HASHTAGS (each is a separate potential place — treat every location hashtag as its own entry):
{json.dumps(segmented_hashtags)}

LOCATION TAG: {location or '(none)'}

PLACES DETECTED IN VIDEO FRAMES:
{json.dumps(video_places)}

HOTELS / STAYS VISIBLE IN VIDEO:
{json.dumps(video_hotels)}

RESTAURANTS / FOOD SPOTS VISIBLE IN VIDEO:
{json.dumps(video_food)}

ACTIVITIES VISIBLE IN VIDEO:
{json.dumps(video_acts)}

ON-SCREEN TEXT FROM VIDEO (OCR):
{json.dumps(ocr_text)}

ITINERARY SLIDES (full text of day-plan / schedule cards visible in video):
{json.dumps(itinerary_slides[:5])}

AUDIO / VOICEOVER CUES:
{json.dumps(audio_cues[:10])}

VIDEO SUMMARY: {video_insights.get('raw_summary','')[:300]}

=== INSTRUCTIONS ===

Extract EVERY distinct place, attraction, food spot, hotel, and activity location.

CRITICAL RULES — read carefully:
1. DO NOT merge similar places. "Vagator Beach" and "North Goa" are SEPARATE entries.
2. DO NOT skip a place just because another nearby place was already included.
3. Every location hashtag (#vagatorbeach, #chapofort, #baga) = one entry each.
4. Every hotel, restaurant, or café mentioned = its own entry.
5. If a place appears in multiple sources (hashtag + video + audio) = still ONE entry, but mark source as "all".
6. Include cities, regions, beaches, forts, temples, cafés, restaurants, hotels, viewpoints, markets — everything.
7. Inferred = true ONLY if you're guessing it's there. Explicitly mentioned = inferred: false.

Return ONLY valid JSON — no markdown:
{{
  "destination": "main city or region of the entire trip",
  "country": "country name",
  "places": [
    {{
      "name": "Exact Place Name (properly capitalised)",
      "type": "city | attraction | food | stay | activity | region",
      "description": "one sentence: what is this place",
      "inferred": false,
      "source": "caption | hashtag | video | audio | all"
    }}
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract travel places. Your job is to find EVERY place — "
                        "never merge or skip. Return ONLY valid JSON. No markdown."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,   # increased — more places need more tokens
            temperature=0.1,   # low temp for consistent extraction
        )
        raw = _strip_fences(resp.choices[0].message.content.strip())
        result = json.loads(raw)

        # ── Safety net: if LLM still returned very few, append raw video places ──
        # This guarantees video-detected places are never silently dropped.
        existing_names = {p["name"].lower() for p in result.get("places", [])}
        extras = []
        for raw_place in video_places + video_hotels + video_food:
            if raw_place.lower() not in existing_names and len(raw_place) > 2:
                extras.append({
                    "name": raw_place,
                    "type": "attraction",
                    "description": "",
                    "inferred": False,
                    "source": "video",
                })
                existing_names.add(raw_place.lower())

        if extras:
            result["places"] = result.get("places", []) + extras
            print(f"  ➕ Added {len(extras)} video-only places not captured by LLM")

        print(f"✅ Places: {len(result.get('places',[]))} extracted for '{result.get('destination')}'")
        return result

    except Exception as e:
        print(f"❌ Place extraction failed: {e}")
        # Fallback: build list directly from all raw sources without LLM
        return _fallback_place_list(
            caption, segmented_hashtags, location,
            video_places, video_hotels, video_food
        )


def _fallback_place_list(caption: str, hashtags: List[str], location: str,
                          video_places: List[str], video_hotels: List[str],
                          video_food: List[str]) -> Dict:
    """
    Emergency fallback: build place list without LLM by combining raw sources.
    Used when the LLM call fails entirely.
    """
    seen  : set         = set()
    places: List[dict]  = []

    def _add(name: str, ptype: str, source: str, inferred: bool = False):
        key = name.lower().strip()
        if key and key not in seen and len(key) > 2:
            seen.add(key)
            places.append({
                "name": name.strip().title(),
                "type": ptype,
                "description": "",
                "inferred": inferred,
                "source": source,
            })

    for h in hashtags:
        _add(h, "attraction", "hashtag")
    for p in video_places:
        _add(p, "attraction", "video")
    for h in video_hotels:
        _add(h, "stay", "video")
    for f in video_food:
        _add(f, "food", "video")
    if location:
        _add(location, "city", "caption")

    return {
        "destination": location or (video_places[0] if video_places else "Unknown"),
        "country": "",
        "places": places,
    }


# ===========================================================================
# HELPERS
# ===========================================================================

def cleanup_temp_files(*paths) -> None:
    """Remove temp files and directories. Non-fatal."""
    import shutil
    for item in paths:
        if item is None:
            continue
        try:
            if isinstance(item, list):
                parent_dirs = set()
                for f in item:
                    if os.path.exists(f):
                        parent_dirs.add(str(Path(f).parent))
                        os.unlink(f)
                for d in parent_dirs:
                    if "tb_frames_" in d:
                        shutil.rmtree(d, ignore_errors=True)
            elif os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
            elif os.path.exists(item):
                os.unlink(item)
        except Exception:
            pass