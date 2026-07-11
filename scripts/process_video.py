"""
Full clip generation pipeline for a single YouTube video.

Efficient two-pass download strategy:
  Pass 1 — Audio only (~40MB): transcribe with faster-whisper, save transcript to disk
  Pass 2 — Per-clip sections only (~15MB each): download just the 30-90s needed per clip

Steps:
  1. Download audio only
  2. Transcribe locally with faster-whisper
  3. Claude reads transcript and picks 10-12 viral clip moments (with timestamps)
  4. For each clip: download just that section at 720p → crop 9:16 → burn subtitles → upload → queue
  5. Mark video as processed

Usage:
  python scripts/process_video.py --video_id VIDEO_ID --url URL --title "Title"
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv
from supabase import create_client
from faster_whisper import WhisperModel
import anthropic

load_dotenv(override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
CHANNEL_ID = os.environ["CHANNEL_ID"]

WHISPER_MODEL = "medium"

LOGO_PATH = str(Path(__file__).parent.parent / "logo.png")


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Download ───────────────────────────────────────────────────────────────────

def _ydl_auth_args():
    args = []
    proxy = os.environ.get("YTDLP_PROXY")
    if proxy:
        args += ["--proxy", proxy]
    cookies = os.environ.get("YOUTUBE_COOKIES_FILE")
    if cookies and os.path.exists(cookies):
        args += ["--cookies", cookies]
    return args


def _ytdlp_client_attempts():
    """
    Ordered (extractor_args, extra_cli_args) pairs to try for YouTube downloads.
    `tv_embedded` needs no cookies/proxy/PO-token at all as of writing (verified
    against a real video: full DASH manifest up to 1080p, real segment download
    confirmed). `web` + cookies/proxy is kept as a fallback in case YouTube
    closes off tv_embedded the way it has closed off other clients before
    (this codebase has been through android, ios, and OAuth2 for the same
    reason) -- so a stale/expired YOUTUBE_COOKIES secret degrades to today's
    known failure mode instead of nothing working at all.
    """
    return [
        ("youtube:player_client=tv_embedded", []),
        ("youtube:player_client=web", _ydl_auth_args()),
    ]


def download_audio_only(url, output_dir):
    """Download just the audio stream — ~40MB for a 40-min podcast."""
    output_path = os.path.join(output_dir, "audio.%(ext)s")
    attempts = _ytdlp_client_attempts()
    last_err = None
    for i, (extractor_args, extra_args) in enumerate(attempts):
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-f", "bestaudio[ext=m4a]/bestaudio/best",
                    "--no-playlist",
                    "--extractor-args", extractor_args,
                    "--js-runtimes", "node",
                    *extra_args,
                    "-o", output_path,
                    url,
                ],
                check=True,
            )
            matches = list(Path(output_dir).glob("audio.*"))
            if matches:
                return str(matches[0])
            last_err = RuntimeError("yt-dlp audio download produced no output file")
        except subprocess.CalledProcessError as e:
            last_err = e
        if i < len(attempts) - 1:
            print(f"    Audio download via {extractor_args} failed, trying next client...")
    raise last_err


def download_full_video(url, output_path):
    """
    Download the full video at the best available quality up to 1080p.
    Uses DASH (video+audio merged) when running locally — 1080p DASH is only
    blocked from datacenter/GitHub Actions IPs, not from a local machine.
    Falls back to 720p then 360p progressive if DASH is unavailable.

    Tries player clients in order (see _ytdlp_client_attempts): tv_embedded
    needs no auth at all as of writing; web+cookies/proxy is the fallback.
    """
    # 720p source is sufficient — ffmpeg scales up to 1080x1920 output.
    # Keeps download ~200MB vs 500MB+ for 1080p on long podcasts.
    format_str = (
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
        "/bestvideo[height<=720]+bestaudio"
        "/22/18"
    )
    attempts = _ytdlp_client_attempts()
    ATTEMPTS_PER_CLIENT = 2
    last_err = None
    for ci, (extractor_args, extra_args) in enumerate(attempts):
        cmd = [
            "yt-dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--socket-timeout", "60",
            "--retries", "10",
            "--fragment-retries", "10",
            "--http-chunk-size", "10M",
            "--extractor-args", extractor_args,
            "--js-runtimes", "node",
            *extra_args,
            "-o", output_path,
            url,
        ]
        for attempt in range(ATTEMPTS_PER_CLIENT):
            if os.path.exists(output_path):
                os.remove(output_path)
            try:
                subprocess.run(cmd, check=True)
                if os.path.exists(output_path) and os.path.getsize(output_path) >= 100_000:
                    return
                last_err = RuntimeError(f"Full video download produced no usable file: {output_path}")
            except subprocess.CalledProcessError as e:
                last_err = e
            is_last_overall = (ci == len(attempts) - 1) and (attempt == ATTEMPTS_PER_CLIENT - 1)
            if not is_last_overall:
                print(f"    Video download via {extractor_args} attempt {attempt + 1} failed, retrying in 30s...")
                time.sleep(30)
    raise last_err


# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe(audio_path):
    """Transcribe audio locally with faster-whisper. Returns (segments, words) with word-level timestamps."""
    print(f"  Loading Whisper {WHISPER_MODEL} model...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("  Transcribing (takes a few minutes for long audio)...")
    raw_segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = []
    words = []
    for s in raw_segments:
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        if s.words:
            for w in s.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
    print(f"  Language: {info.language} ({info.language_probability:.0%} confidence)")
    return segments, words


# ── Claude clip selection ──────────────────────────────────────────────────────

def _classify_hook_type(hook: str) -> str:
    """Classify a hook into a type using simple heuristics (no API call)."""
    h = hook.lower().strip()
    if h.endswith("?"):
        return "question"
    if any(c in h for c in ["$", "¢", "£", "€", "%"]) or any(
        w in h for w in ["million", "thousand", "cedis", "dollars", "percent", "years", "months"]
    ):
        return "stat"
    if any(w in h for w in ["i was wrong", "i failed", "i lied", "i never", "i lost", "i made a mistake", "i regret"]):
        return "confession"
    if any(w in h for w in ["when i was", "the day i", "one day", "back when", "growing up", "i remember when"]):
        return "story"
    if any(w in h for w in ["stop ", "never ", "don't ", "you should", "you must", "the truth is", "nobody tells"]):
        return "advice"
    if any(w in h for w in ["god", "church", "jesus", "bible", "pastor", "prayer", "faith", "christian"]):
        return "controversy"
    if h.startswith("i ") or h.startswith("i'"):
        return "provocative"
    return "statement"


def _classify_topic(hook: str) -> str:
    """Classify a clip's topic from its hook using keyword matching."""
    h = hook.lower()
    if any(w in h for w in ["money", "rich", "broke", "income", "salary", "paid", "cedis", "invest", "wealth", "profit", "revenue", "afford"]):
        return "money"
    if any(w in h for w in ["business", "entrepreneur", "startup", "company", "brand", "client", "customer", "market", "product", "sell"]):
        return "business"
    if any(w in h for w in ["married", "wife", "husband", "relationship", "love", "date", "family", "children", "divorce", "partner"]):
        return "relationships"
    if any(w in h for w in ["god", "church", "jesus", "pray", "pastor", "faith", "christian", "gospel", "worship"]):
        return "faith"
    if any(w in h for w in ["school", "university", "degree", "education", "student", "teacher", "learn"]):
        return "education"
    if any(w in h for w in ["africa", "ghana", "nigerian", "kenyan", "continent", "accra", "lagos"]):
        return "africa"
    if any(w in h for w in ["discipline", "motivation", "success", "failure", "hustle", "grind", "mindset", "habit", "goal"]):
        return "mindset"
    return "personal"


def _extract_clip_transcript(segments: list, start_s: float, end_s: float) -> str:
    """Return the spoken text for transcript segments overlapping [start_s, end_s]."""
    parts = []
    for seg in segments:
        if seg.get("end", 0) > start_s and seg.get("start", 0) < end_s:
            parts.append(seg["text"].strip())
    return " ".join(parts)[:1200]


def _clip_variant(video_id: str, clip_index: int) -> str:
    """
    Deterministic, stateless 50/50 A/B bucket for caption/zoom style ("clean" vs
    "dynamic" -- see cut_and_subtitle). Pure function of (video_id, clip_index)
    so no assignment needs to be threaded through or persisted anywhere except
    the log used for future correlation with real view performance.
    """
    import hashlib
    digest = hashlib.md5(f"{video_id}:{clip_index}".encode()).hexdigest()
    return "dynamic" if int(digest, 16) % 2 else "clean"


def _log_clip_selections(supabase, video_id: str, clips: list, segments: list | None = None) -> None:
    """Log clip selections with hook type, topic, and transcript text for content learning."""
    try:
        rows = []
        for i, clip in enumerate(clips):
            hook     = clip.get("hook", "")
            start_s  = clip.get("start_seconds") or 0
            end_s    = clip.get("end_seconds") or 0
            duration = round(end_s - start_s) if end_s > start_s else None
            transcript = _extract_clip_transcript(segments, start_s, end_s) if segments else None
            rows.append({
                "video_id":         video_id,
                "clip_index":       i,
                "hook":             hook[:200],
                "duration_seconds": duration,
                "hook_type":        _classify_hook_type(hook),
                "topic_category":   _classify_topic(hook),
                "clip_transcript":  transcript,
                "caption_variant":  _clip_variant(video_id, i),
            })
        try:
            supabase.table("clip_selection_log").upsert(rows, on_conflict="video_id,clip_index").execute()
            print(f"  [Memory] Logged {len(rows)} clip selections (with transcripts).")
        except Exception:
            try:
                # caption_variant column may not exist yet -- retry without it
                rows_no_variant = [{k: v for k, v in r.items() if k != "caption_variant"} for r in rows]
                supabase.table("clip_selection_log").upsert(rows_no_variant, on_conflict="video_id,clip_index").execute()
                print(f"  [Memory] Logged {len(rows)} clips (add caption_variant TEXT column to enable style A/B analysis).")
            except Exception:
                # clip_transcript column may not exist either -- retry without both
                rows_basic = [{k: v for k, v in r.items() if k not in ("clip_transcript", "caption_variant")} for r in rows]
                supabase.table("clip_selection_log").upsert(rows_basic, on_conflict="video_id,clip_index").execute()
                print(f"  [Memory] Logged {len(rows)} clips (add clip_transcript/caption_variant columns to enable content analysis).")
    except Exception as e:
        print(f"  [Memory] Could not log selections: {e}")


def _load_channel_intelligence(supabase) -> str:
    """Build explicit ranked directives from real channel performance data and algorithm research."""
    try:
        stats_row = supabase.table("channel_intelligence").select("stats").eq("id", "singleton").maybe_single().execute()
        log = supabase.table("clip_selection_log").select("hook_type,topic_category,performance_tier,views").not_("performance_tier", "is", None).execute()
        algo_row = supabase.table("settings").select("value").eq("key", "algorithm_research_v2").maybe_single().execute()

        parts = []

        if stats_row.data and stats_row.data.get("stats"):
            s = stats_row.data["stats"]
            parts.append(f"Channel data based on {s.get('clips_analysed','?')} clips posted and {s.get('total_views','?')} total views.")

        if log.data:
            def build_map(key):
                m = {}
                for r in log.data:
                    k = (r.get(key) or "unknown").lower()
                    if k not in m:
                        m[k] = {"views": [], "top": 0, "total": 0}
                    if r.get("views"):
                        m[k]["views"].append(r["views"])
                    m[k]["total"] += 1
                    if r.get("performance_tier") == "top":
                        m[k]["top"] += 1
                return sorted(
                    [(k, round(sum(v["views"]) / len(v["views"])) if v["views"] else 0, v["top"], v["total"])
                     for k, v in m.items()],
                    key=lambda x: -x[1]
                )

            hook_ranks = build_map("hook_type")
            topic_ranks = build_map("topic_category")
            top_avg = hook_ranks[0][1] if hook_ranks else 1

            if hook_ranks:
                lines = ["HOOK TYPE PERFORMANCE (real avg views on this channel):"]
                for i, (name, avg, top, total) in enumerate(hook_ranks):
                    top_pct = round(top / total * 100) if total else 0
                    if i == 0:
                        directive = "PRIORITISE — best performer"
                    elif i < 3:
                        directive = "PREFER"
                    elif avg < top_avg * 0.5:
                        directive = "AVOID unless exceptional"
                    else:
                        directive = "NEUTRAL"
                    lines.append(f"  {i+1}. {name.capitalize()}: {avg} avg views, {top_pct}% top tier — {directive}")
                parts.append("\n".join(lines))

            if topic_ranks:
                lines = ["TOPIC CATEGORY PERFORMANCE (real avg views on this channel):"]
                for i, (name, avg, top, total) in enumerate(topic_ranks[:8]):
                    top_pct = round(top / total * 100) if total else 0
                    directive = "PRIORITISE" if i == 0 else ("PREFER" if i < 3 else "NEUTRAL")
                    lines.append(f"  {i+1}. {name.capitalize()}: {avg} avg views, {top_pct}% top tier — {directive}")
                parts.append("\n".join(lines))

        if algo_row and algo_row.data:
            algo_text = (algo_row.data.get("value") or {}).get("text", "")
            if algo_text:
                parts.append(f"PLATFORM ALGORITHM INSIGHTS (current, from web research):\n{algo_text}")

        return "\n\n".join(parts) if parts else ""
    except Exception:
        pass
    return ""


def select_clips(anthropic_client, segments, video_title, supabase=None):
    """Ask Claude to pick 10-12 viral moments from the transcript."""
    lines = []
    any_signals = False
    for seg in segments:
        sm, ss = int(seg["start"] // 60), int(seg["start"] % 60)
        em, es = int(seg["end"] // 60), int(seg["end"] % 60)
        tag = seg.get("signal_tag")
        suffix = f"  [{tag}]" if tag else ""
        any_signals = any_signals or bool(tag)
        lines.append(f"[{sm:02d}:{ss:02d}-{em:02d}:{es:02d}] {seg['text']}{suffix}")

    transcript_text = "\n".join(lines)
    if len(transcript_text) > 46000:
        transcript_text = transcript_text[:46000] + "\n...[truncated]"

    signal_legend = ""
    if any_signals:
        signal_legend = """
━━━ VISUAL/AUDIO SIGNAL TAGS (if present) ━━━
Some lines end with a bracketed tag like [F1 M:hi C:1 A:hi] -- this is computed
automatically from the raw video/audio, not from you:
  F<n>  = faces visible in that moment (0, 1, or 2+ = two or more)
  M:lo/med/hi = visual motion/energy, relative to THIS video's own baseline
  C:<n> = number of hard camera cuts in that moment
  A:lo/med/hi/silence = audio energy, relative to THIS video's own baseline
    ("silence" flags a notable quiet/pause stretch)

These are an ADDITIONAL, SOFT signal -- never a hard filter. A quiet, still,
powerful confession can and should still be picked over a loud, high-motion
moment with weak content. Use these tags mainly to:
  (a) break ties between two similarly-scored candidates -- prefer the one
      that is visually/aurally alive, and
  (b) catch cases where a transcript-strong moment sits on a dead, static, or
      badly mid-cut shot that would undercut it on video.
Segments with no tag simply had no signal computed for them -- treat them
exactly as you would treat any line, on transcript content alone.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    intelligence = _load_channel_intelligence(supabase) if supabase else ""
    intelligence_block = ""
    if intelligence:
        intelligence_block = f"""

━━━ CHANNEL PERFORMANCE DATA — MANDATORY DIRECTIVES ━━━

{intelligence}

These rankings come from real views on clips already posted on this channel. They are not suggestions — they are your selection criteria. When two candidate moments are of similar quality, always choose the one whose hook type or topic category ranks higher above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print("  [Intelligence] Loaded channel performance directives into prompt.")
    else:
        print("  [Intelligence] No brief available yet -- using base rules only.")

    prompt = f"""You are an expert viral short-form content strategist who deeply understands what makes people stop scrolling and watch a video clip all the way through to completion.

Your job: analyze the provided timestamped podcast transcript and identify EXACTLY 7 of the highest-impact moments to clip for Instagram Reels, TikTok, YouTube Shorts, and Facebook Reels. Quality over quantity — 7 exceptional clips beat 12 average ones.{intelligence_block}

Video title: {video_title}
{signal_legend}
Transcript (with timestamps):
Each line shows [MM:SS-MM:SS] start-end time for that segment, followed by the spoken text.
{transcript_text}

━━━ CLIP SELECTION RULES ━━━

1. CHRONOLOGICAL SYSTEMATIC REVIEW — Read through the entire transcript from start to finish. Do not ignore the middle sections. Distribute your selections evenly across the entire runtime. Do not cluster clips in a single high-energy segment. IMPORTANT: The opening 2–3 minutes of a podcast is typically an intro montage — rapid jump cuts between unrelated moments designed to tease the full episode. These make no sense as standalone clips. Use your judgment to identify and skip this intro section; look for where the hosts settle into the actual conversation (usually when they introduce themselves or the guest properly) and start your selections from there.

2. TARGET LENGTH — Target 45 to 65 seconds per clip. This is the proven sweet spot for completion rate across all 4 platforms. Never go below 30 seconds or above 90 seconds. Tighter is better. The clip must start precisely on the opening word of the hook and end the moment the idea concludes — do not trail off into the next topic or filler.

3. THE SCROLL-STOP TEST — Before selecting any clip, apply this test to its opening sentence: imagine someone in a noisy room, half-watching their phone, thumb already moving to swipe. Would they freeze on THIS specific sentence? If the answer is anything less than "yes, immediately" — reject it and find a better moment. Generic wisdom fails this test. Setup sentences fail this test. Anything that needs context to land fails this test.

  A hook that passes the test is one of these:
  • A specific number or amount that surprises ("She lost 300,000 cedis in one month")
  • A direct personal confession or failure ("I almost destroyed my own company doing this")
  • A statement that makes the audience choose a side ("Most African entrepreneurs will never scale and it is their own fault")
  • A counter-intuitive reversal of something everyone believes ("Working harder is exactly what keeps you broke")
  • A story opening with immediate tension ("The day I fired my best friend was the day the business started growing")
  • A question with an answer the viewer desperately wants ("Why do the most disciplined people still fail?")

  AUTOMATIC REJECTION — Never start a clip on:
  • Filler words: "So...", "Um...", "You know...", "Like I said...", "Basically..."
  • Slow context: "Today we're going to talk about...", "I want to tell you something about..."
  • Generic advice that could apply to anyone: "You need to believe in yourself", "Hard work pays off"
  • A mid-thought that requires the previous sentence to make sense
  • A compliment or pleasantry: "That's a great question", "Absolutely, I agree"

4. THE STRONG ENDING TEST — A clip is only as good as its ending. Before finalising any clip, apply this test to its closing moment: would a viewer who just finished watching feel satisfied, provoked, or compelled to rewatch?

  A strong ending is one of these:
  • A punchline or twist as a final statement — the idea lands and then stops
  • A direct challenge to the audience ("That is the truth and you know it")
  • A specific result revealed at the end ("...and we made 40,000 cedis that month")
  • A counter-intuitive conclusion that closes the thought completely
  • A quiet moment of silence after a hard truth — the clip ends right before the next breath

  A weak ending sounds like:
  • Trailing off: "...you know, it just kind of... yeah"
  • Bleeding into the next topic: "Anyway, what I was going to say about that is..."
  • Over-explaining: "...what I mean is basically what I was trying to say is..."
  • An incomplete sentence or half-finished thought

  Your end_seconds MUST land on the last word of the strong closing beat. Cut immediately after. Do not let the clip run into the next sentence or the speaker's next breath.

5. CONTENT COHESION — Each clip must contain ONE complete self-contained idea with a 3-part arc:
  • HOOK (0-5s): The opening line creates tension, curiosity, or surprise in the viewer's mind
  • DEVELOPMENT (middle): The speaker builds the idea — adds a specific detail, tells the story, or explains the stakes
  • PAYOFF (final 5-10s): The clip ends on a clear answer, revelation, punchline, or strong closing statement

  A clip missing the Payoff will not be rewatched or shared no matter how strong the hook is. Reject any clip where the idea is not fully resolved within the selected time range.

  Also prioritise moments that are:
  • A specific number, amount, or statistic that reveals something surprising
  • A personal failure, mistake, or lesson the speaker learned the hard way
  • A take that the Konnected Minds audience (Ghana/Africa, entrepreneurship, business, faith, ambition) will passionately agree or disagree with
  • A revelation that reframes how the audience thinks about something they already believe

6. ALGORITHM SIGNALS — Pick moments that will drive measurable actions:
  • Comments: the audience needs to argue about it, share their own story, or tag someone
  • Shares: the clip must feel like "I need to send this to someone right now"
  • Saves: practical insight or a hard truth people want to return to
  • Replays: a punchline, a stat, or a twist delivered so well they want to hear it again

7. TIMESTAMP CONVERSION — Each transcript line shows [MM:SS-MM:SS] format. Convert the start time of your chosen clip to raw integer seconds for start_seconds, and the end time to raw integer seconds for end_seconds. Example: a clip starting at 02:05 and ending at 03:07 becomes start_seconds: 125, end_seconds: 187.

━━━ SELECTION PROCESS — Follow these steps before writing any JSON ━━━

STEP 1 — CANDIDATE SCAN: Read the full transcript and identify 12 to 15 moments that could potentially make strong clips. Note the timestamp and opening line for each.

STEP 2 — SCORE EACH CANDIDATE: For every candidate, mentally assign three scores:
  - Hook score (0-10): How immediately compelling is the first sentence to someone scrolling?
  - Payoff score (0-10): Does the clip land on a strong, satisfying conclusion?
  - Audience score (0-10): Will the KonnectedMinds audience (Ghana/Africa, entrepreneurship, business, ambition) feel this deeply?
  Total = Hook + Payoff + Audience (max 30). Reject any candidate scoring below 21.

STEP 3 — APPLY CHANNEL DATA AND SIGNALS: Cross-reference your scored candidates against the performance directives above. Where two candidates have similar scores, choose the one whose hook type or topic category ranks higher in the channel data. Where present, also weigh the visual/audio signal tags per the rules above as a further tiebreaker -- never as a rejection criterion on their own.

STEP 4 — SELECT EXACTLY 7: Take the 7 highest-scoring candidates after Steps 2 and 3. These are your final clips. Do not include any clip you are not confident about.

━━━ CAPTION RULES ━━━

NEVER use em dashes (—) anywhere in any caption. Use a comma, a full stop, or rewrite the sentence instead.

Write platform-specific captions for each clip. They must sound like a real person wrote them, not a brand or marketing team. Avoid hype words like "game-changer", "powerful", "incredible". Write like you are texting a friend about something that genuinely surprised you.

instagram: 2–3 short sentences. Expand on the hook and make them want to watch. End with 5–8 relevant hashtags on a new line. Tone: direct and real, like a sharp comment not an ad.

tiktok: 1–2 very casual sentences. Sounds like someone who just watched this and had to share it. 3–5 hashtags max. Tone: off the cuff, human, slightly opinionated.

youtube: A standalone title that works as a YouTube Shorts title (max 80 characters). Create curiosity or promise a specific payoff. No hashtags. Capitalise like a headline but keep it plain spoken.

facebook: 2–3 sentences for a slightly older audience. More context, less hype. Write like you are sharing something interesting you came across. No hashtags.

━━━ OUTPUT FORMAT ━━━

CRITICAL: Output ONLY the raw JSON array containing EXACTLY 7 clips. Do not include any introductory sentences, conversational text, markdown code fences, or concluding remarks. The response must start with [ and end with ] and be 100% pure parseable JSON.

[
  {{
    "start_seconds": 125,
    "end_seconds": 187,
    "hook": "The exact first sentence spoken that opens this clip",
    "captions": {{
      "instagram": "Caption text here.\\n\\n#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5",
      "tiktok": "casual caption here #hashtag1 #hashtag2 #hashtag3",
      "youtube": "YouTube Shorts Title That Creates Curiosity",
      "facebook": "Facebook caption here that gives a bit more context."
    }}
  }}
]"""

    raw = ""
    for attempt in range(4):
        try:
            # Stream the response -- keeps connection alive on large prompts
            with anthropic_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=6000,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                raw = stream.get_final_text()
            break
        except Exception as e:
            if attempt == 3:
                raise
            wait = 15 * (attempt + 1)
            print(f"  Claude API attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
            import time; time.sleep(wait)

    raw = raw.strip()
    # Robust extraction: find the outermost JSON array regardless of any surrounding text
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx + 1]

    clips = json.loads(raw)
    valid = []
    for c in clips:
        duration = c["end_seconds"] - c["start_seconds"]
        if not (28 <= duration <= 95):
            continue
        if "caption" in c and "captions" not in c:
            c["captions"] = {p: c["caption"] for p in ["instagram", "tiktok", "youtube", "facebook"]}
        valid.append(c)
    return valid


# ── Segment signal analysis (cheap, local, free) ────────────────────────────────
# Gives clip selection eyes and ears without any paid API calls: computes coarse
# per-window visual/audio energy from the raw video/audio using the same cv2/
# mediapipe already installed for face-crop tracking, then folds it into the
# transcript as compact tags Claude can weigh alongside the words.

def _frame_diffs(frames: list) -> list:
    """Mean absolute pixel difference between consecutive downscaled grayscale frames."""
    import cv2
    import numpy as np
    thumbs = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        for f in frames
    ]
    return [float(np.mean(np.abs(thumbs[i] - thumbs[i - 1]))) for i in range(1, len(thumbs))]


def detect_shot_cuts(frames: list, threshold: float = 40.0) -> int:
    """
    Count hard camera cuts among a list of already-decoded frames via frame-diff
    thresholding. Takes frames rather than a path/timestamps so callers that
    already have frames in hand (crop tracking, signal windows) can reuse it
    without re-decoding the video.
    """
    return sum(1 for d in _frame_diffs(frames) if d > threshold)


def _extract_audio_pcm(source_path: str, sr: int = 16000):
    """Demux mono PCM from any audio/video file via ffmpeg. Returns (int16 ndarray, sr)."""
    import numpy as np
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", source_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16), sr


def _audio_window_stats(pcm, sr: int, start_s: float, end_s: float) -> dict:
    """RMS / silence-ratio for one time window of already-extracted PCM."""
    import numpy as np
    i0, i1 = max(0, int(start_s * sr)), min(len(pcm), int(end_s * sr))
    if i1 <= i0:
        return {"rms": 0.0, "silence_ratio": 1.0}
    window = pcm[i0:i1].astype(np.float32)
    silence_thresh = 0.02 * 32768  # 2% of int16 full scale
    return {
        "rms": float(np.sqrt(np.mean(window ** 2))),
        "silence_ratio": float(np.mean(np.abs(window) < silence_thresh)),
    }


def _compute_segment_signals(video_path: str, segments: list, audio_source: str = None) -> list:
    """
    Returns new segment dicts (shallow copies of `segments`) each with an added
    'signal_tag' string summarizing local visual/audio energy for the ~15s
    window(s) overlapping that segment -- computed once per video from cheap
    local cv2/mediapipe/ffmpeg/numpy analysis. No paid API calls.

    Never raises: internal stages degrade independently (missing face detector,
    failed audio extraction); total failure returns `segments` unchanged so
    callers always have a safe transcript-only fallback.
    """
    import cv2
    import numpy as np

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps else 0.0
        if duration <= 0:
            cap.release()
            raise RuntimeError("could not determine duration")

        WINDOW_SECONDS, MAX_WINDOWS, SAMPLES_PER_WINDOW = 15.0, 300, 4
        window_seconds = WINDOW_SECONDS
        n_windows = max(1, int(duration // window_seconds) + 1)
        if n_windows > MAX_WINDOWS:
            window_seconds = duration / MAX_WINDOWS
            n_windows = MAX_WINDOWS

        mp_detector = None
        haar_cascades = []
        try:
            import mediapipe as mp
            mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.25,
            )
        except Exception:
            for xml in ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]:
                try:
                    c = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
                    if not c.empty():
                        haar_cascades.append(c)
                except Exception:
                    pass

        window_stats = []
        try:
            for w in range(n_windows):
                w_start = w * window_seconds
                w_end = min(duration, w_start + window_seconds)
                frames, face_count = [], 0
                for s in range(SAMPLES_PER_WINDOW):
                    t = w_start + (s + 0.5) * (w_end - w_start) / SAMPLES_PER_WINDOW
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frames.append(frame)
                    try:
                        if mp_detector is not None:
                            res = mp_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            face_count = max(face_count, len(res.detections) if res.detections else 0)
                        elif haar_cascades:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            for cascade in haar_cascades:
                                faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
                                if len(faces):
                                    face_count = max(face_count, len(faces))
                                    break
                    except Exception:
                        pass
                window_stats.append({
                    "start": w_start, "end": w_end, "face_count": face_count,
                    "motion": float(np.mean(_frame_diffs(frames))) if len(frames) >= 2 else 0.0,
                    "cuts": detect_shot_cuts(frames) if len(frames) >= 2 else 0,
                })
        finally:
            if mp_detector is not None:
                try:
                    mp_detector.close()
                except Exception:
                    pass
            cap.release()

        try:
            pcm, sr = _extract_audio_pcm(audio_source or video_path)
            for ws in window_stats:
                ws.update(_audio_window_stats(pcm, sr, ws["start"], ws["end"]))
        except Exception as e:
            print(f"  [Signals] audio extraction failed ({e}) -- visual signals only")
            for ws in window_stats:
                ws["rms"], ws["silence_ratio"] = 0.0, None

        # Tier relative to THIS video's own distribution -- absolute scale is meaningless
        # across a slow talking-head episode vs a fast-cut highlight reel.
        def _tier(value, values):
            if not values or value <= 0:
                return "lo"
            lo_cut, hi_cut = np.percentile(values, [33, 66])
            return "hi" if value >= hi_cut else ("med" if value >= lo_cut else "lo")

        motions = [w["motion"] for w in window_stats if w["motion"] > 0]
        rmses = [w["rms"] for w in window_stats if w["silence_ratio"] is not None and w["rms"] > 0]
        for ws in window_stats:
            ws["motion_tier"] = _tier(ws["motion"], motions)
            if ws["silence_ratio"] is None:
                ws["audio_tier"] = None
            elif ws["silence_ratio"] > 0.85:
                ws["audio_tier"] = "silence"
            else:
                ws["audio_tier"] = _tier(ws["rms"], rmses)

        TIER_RANK = {"lo": 0, "med": 1, "hi": 2, "silence": 0, None: -1}

        def _tag(seg_start, seg_end):
            overlapping = [w for w in window_stats if w["end"] > seg_start and w["start"] < seg_end]
            if not overlapping:
                return None
            face_count = max(w["face_count"] for w in overlapping)
            cuts = max(w["cuts"] for w in overlapping)
            motion_tier = max((w["motion_tier"] for w in overlapping), key=lambda t: TIER_RANK[t])
            audio_tiers = [w["audio_tier"] for w in overlapping if w["audio_tier"]]
            audio_tier = max(audio_tiers, key=lambda t: TIER_RANK[t]) if audio_tiers else None
            parts = [f"F{face_count}", f"M:{motion_tier}"]
            if cuts:
                parts.append(f"C:{cuts}")
            if audio_tier:
                parts.append(f"A:{audio_tier}")
            return " ".join(parts)

        out = []
        for seg in segments:
            new_seg = dict(seg)
            tag = _tag(seg.get("start", 0), seg.get("end", 0))
            if tag:
                new_seg["signal_tag"] = tag
            out.append(new_seg)
        return out

    except Exception as e:
        print(f"  [Signals] computation failed ({e}) -- selecting from transcript only")
        return segments


# ── Video processing ───────────────────────────────────────────────────────────

def _compute_speaker_crop_path(video_path: str, start_s: float, duration: float, fps: float) -> list:
    """
    Compute a per-frame (x, y, crop_w, crop_h) path tracking the dominant speaker.

    Design:
    - Tight 9:16 crop: full source height, horizontal tracking only.
      For a 720p (1280x720) source this is 405x720 -- shows face + upper body,
      fills 1080x1920 with no wasted bars.
    - MediaPipe face detection every DETECT_EVERY output frames (~10fps at 30fps).
    - Linear interpolation between detections for sub-detection-interval smoothness.
    - EMA smoothing (alpha=0.08) makes the crop move like a deliberate camera
      operator, not an AI twitching every second.
    - Hard velocity cap (1.5% of frame width per frame) prevents jarring pans.
    - Haar cascade fallback if MediaPipe is unavailable.
    - Pixel-variance side fallback if zero faces are ever detected (B-roll, graphics).
    - Shot-aware: detects hard camera cuts within the clip (via _frame_diffs) and
      re-clusters the dominant speaker independently per shot, resetting the EMA/
      velocity cap at each cut -- a mid-clip camera-angle change no longer drags a
      stale locked position into a shot it doesn't match.
    - All errors fall back silently to a static center crop -- never raises.
    """
    import cv2
    import numpy as np

    n_frames = max(1, int(round(duration * fps)))

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 9:16 portrait crop -- full source height, partial width
        crop_h = frame_h
        crop_w = int(round(frame_h * 9 / 16))
        if crop_w > frame_w:          # ultra-wide or portrait source
            crop_w = frame_w
            crop_h = int(round(frame_w * 16 / 9))
        max_x = frame_w - crop_w
        y     = (frame_h - crop_h) // 2   # 0 for standard 16:9 sources

        # MediaPipe setup
        mp_detector = None
        try:
            import mediapipe as mp
            mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,             # full-range model, handles far faces
                min_detection_confidence=0.25,
            )
        except Exception:
            pass

        # Haar cascade fallback setup (only when MediaPipe unavailable)
        haar_cascades = []
        if mp_detector is None:
            for xml in ["haarcascade_frontalface_default.xml",
                        "haarcascade_frontalface_alt2.xml"]:
                try:
                    c = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
                    if not c.empty():
                        haar_cascades.append(c)
                except Exception:
                    pass

        # Detect every DETECT_EVERY output frames (~10fps at 30fps source)
        DETECT_EVERY = 3
        start_frame  = int(round(start_s * src_fps))
        detections   = {}   # output_frame_idx -> cx (face center x, source pixels)
        frames_for_variance = []

        for out_fi in range(0, n_frames, DETECT_EVERY):
            # Map output frame index to source frame (accounts for fps mismatch)
            src_fi = int(round(out_fi * src_fps / fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + src_fi)
            ret, frame = cap.read()
            if not ret:
                break

            frames_for_variance.append(frame)
            cx_best = None

            if mp_detector is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_detector.process(rgb)
                if res.detections:
                    # Dominant face: highest confidence x relative area product
                    best_score = -1.0
                    for det in res.detections:
                        bb    = det.location_data.relative_bounding_box
                        score = det.score[0] * bb.width * bb.height
                        if score > best_score:
                            best_score = score
                            cx_best    = int((bb.xmin + bb.width * 0.5) * frame_w)
            else:
                gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                min_sz = max(30, frame_w // 15)
                for cascade in haar_cascades:
                    faces = cascade.detectMultiScale(gray, 1.1, 3,
                                                     minSize=(min_sz, min_sz))
                    if len(faces):
                        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                        cx_best = fx + fw // 2
                        break

            if cx_best is not None:
                detections[out_fi] = cx_best

        if mp_detector:
            try:
                mp_detector.close()
            except Exception:
                pass
        cap.release()

        n_windows = max(1, n_frames // DETECT_EVERY)
        print(f"  [FaceTrack] {len(detections)}/{n_windows} windows have faces  "
              f"(crop {crop_w}x{crop_h} from {frame_w}x{frame_h})")

        # ── Detect shot cuts within the sampled sequence ──────────────────────
        # A hard camera-angle change mid-clip invalidates a single clip-wide
        # dominant-speaker lock (see below) -- segment into shots first so each
        # gets its own independent lock instead of dragging a stale position
        # into a mismatched camera angle. Reduces to today's behaviour exactly
        # when there are no cuts (the common case -- most clips are one shot).
        SHOT_CUT_THRESHOLD = 40.0
        sample_diffs = _frame_diffs(frames_for_variance) if len(frames_for_variance) >= 2 else []
        # sample_diffs[k] is the diff between sampled frame k and k+1 (output-frame
        # indices k*DETECT_EVERY and (k+1)*DETECT_EVERY); a cut there means the new
        # shot starts at output frame (k+1)*DETECT_EVERY.
        shot_starts = sorted({0} | {
            (k + 1) * DETECT_EVERY for k, d in enumerate(sample_diffs) if d > SHOT_CUT_THRESHOLD
        })
        shot_starts = [s for s in shot_starts if s < n_frames]
        shot_bounds = list(zip(shot_starts, shot_starts[1:] + [n_frames]))
        if len(shot_bounds) > 1:
            print(f"  [FaceTrack] {len(shot_bounds)} shots detected in this clip "
                  f"(cuts at {shot_starts[1:]})")

        # ── Cluster to dominant speaker, independently per shot ───────────────
        # In a two-person podcast MediaPipe alternates between left and right faces.
        # Split detections by which horizontal half they land in, pick the majority.
        # Locks the crop to ONE speaker per shot (not per clip) so a mid-clip
        # camera-angle change gets its own lock instead of inheriting a stale one.
        raw_cx = np.full(n_frames, float(frame_w // 2), dtype=np.float64)
        mid = frame_w / 2

        for shot_start, shot_end in shot_bounds:
            shot_dets = {fi: cx for fi, cx in detections.items() if shot_start <= fi < shot_end}

            if shot_dets:
                left_dets  = {fi: cx for fi, cx in shot_dets.items() if cx <  mid}
                right_dets = {fi: cx for fi, cx in shot_dets.items() if cx >= mid}
                primary    = left_dets if len(left_dets) >= len(right_dets) else right_dets
                dom_side   = "left"    if len(left_dets) >= len(right_dets) else "right"
                print(f"  [FaceTrack] shot [{shot_start}:{shot_end}) dominant speaker: {dom_side}  "
                      f"({len(primary)}/{len(shot_dets)} detections kept)")

                keys    = sorted(primary.keys())
                last_cx = float(primary[keys[0]])
                det_map = {fi: float(primary[fi]) for fi in keys}
                for f in range(shot_start, shot_end):
                    if f in det_map:
                        last_cx = det_map[f]
                    raw_cx[f] = last_cx
                raw_cx[shot_start:keys[0]] = float(primary[keys[0]])

            elif frames_for_variance:
                # Zero face detections in this shot -- pixel-variance picks left vs right
                shot_frames = [
                    frames_for_variance[fi // DETECT_EVERY]
                    for fi in range(shot_start, shot_end, DETECT_EVERY)
                    if fi // DETECT_EVERY < len(frames_for_variance)
                ]
                if shot_frames:
                    third     = frame_w // 3
                    left_var  = float(np.var(np.array(
                        [f[:, :third] for f in shot_frames], dtype=np.float32)))
                    right_var = float(np.var(np.array(
                        [f[:, 2 * third:] for f in shot_frames], dtype=np.float32)))
                    dom_cx    = third // 2 if left_var > right_var else 2 * third + third // 2
                    raw_cx[shot_start:shot_end] = float(dom_cx)
                    print(f"  [FaceTrack] shot [{shot_start}:{shot_end}) no faces -- variance fallback -> "
                          f"{'left' if left_var > right_var else 'right'} side (cx={dom_cx})")

        # ── EMA smoothing: alpha=0.04 ~ 25-frame lag ──────────────────────────
        # Low alpha = locked/stable feel for static podcast shots.
        # The crop barely moves; only follows large, sustained head shifts.
        # Hard reset (no blend) at each shot boundary -- a real cut should snap
        # instantly, not slowly pan across two unrelated shots.
        ALPHA           = 0.04
        shot_starts_set = set(shot_starts)
        smooth          = np.empty(n_frames, dtype=np.float64)
        smooth[0]       = raw_cx[0]
        for f in range(1, n_frames):
            if f in shot_starts_set:
                smooth[f] = raw_cx[f]
            else:
                smooth[f] = ALPHA * raw_cx[f] + (1.0 - ALPHA) * smooth[f - 1]

        # ── Velocity cap: max 0.4% of frame width per frame ───────────────────
        # ~5px per frame at 1280px wide = very gentle drift, never a pan.
        # Skipped exactly at a shot boundary -- an instant jump there is a real
        # cut, not a jarring pan to be capped.
        max_vel = frame_w * 0.004
        for f in range(1, n_frames):
            if f in shot_starts_set:
                continue
            delta = smooth[f] - smooth[f - 1]
            if abs(delta) > max_vel:
                smooth[f] = smooth[f - 1] + (max_vel if delta > 0 else -max_vel)

        # Convert smoothed center-x to crop boxes
        result = []
        for cx in smooth:
            x = int(round(float(cx))) - crop_w // 2
            x = max(0, min(x, max_x))
            result.append((x, y, crop_w, crop_h))
        return result

    except Exception as e:
        print(f"  [FaceTrack] error: {e} -- center-crop fallback")
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            fw   = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            fh   = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
        except Exception:
            fw, fh = 1280, 720
        ch = fh
        cw = min(int(round(fh * 9 / 16)), fw)
        if cw == fw:
            ch = int(round(fw * 16 / 9))
        x = (fw - cw) // 2
        y = (fh - ch) // 2
        return [(x, y, cw, ch)] * n_frames


def cut_and_subtitle(section_path, offset_seconds, duration, words, output_path, clip_idx, tmpdir, chunk_size=3, hook="", variant="clean"):
    """
    Render a single clip with blur-background 9:16 framing.

    Pipeline:
      1. Extract raw frames from source via ffmpeg
      2. Per frame: scale source to cover 1080x1920 + blur (background),
         scale source to fit width (foreground, horizontally centered on the
         tracked dominant speaker), center fg on blurred bg
      3. Overlay logo and subtitles (style depends on `variant`)
      4. Encode frames + audio in a single pass

    `variant`: "clean" (default) = white-box captions, no zoom, matching the
    established brand look. "dynamic" = karaoke word-highlight captions +
    slow Ken Burns zoom-in, for A/B comparison against real view performance.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    W, H = 1080, 1920

    # Probe source FPS
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", section_path],
        capture_output=True, text=True,
    )
    try:
        num, den = probe.stdout.strip().split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0

    # Tracked dominant-speaker x-path (source pixel space) -- replaces a static
    # center crop so the foreground follows whoever is actually on screen.
    # Never blocks rendering: falls back to None (static center) on any error.
    try:
        crop_path = _compute_speaker_crop_path(section_path, offset_seconds, duration, fps)
    except Exception as e:
        print(f"  [FaceTrack] crop path unavailable ({e}) -- static center fallback")
        crop_path = None

    # Audio: cut low rumble, boost voice presence, normalise to -14 LUFS
    audio_filter = (
        "highpass=f=80,"
        "equalizer=f=300:width_type=o:width=1:g=-3,"
        "equalizer=f=3000:width_type=o:width=1.5:g=3,"
        "loudnorm=I=-14:TP=-1:LRA=11"
    )

    # ── 1. Extract raw frames ────────────────────────────────────────────────
    frames_dir = os.path.join(tmpdir, f"frames_{clip_idx}")
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg",
        "-ss", str(offset_seconds), "-i", section_path,
        "-t", str(duration),
        os.path.join(frames_dir, "frame_%06d.png"), "-y",
    ], check=True, capture_output=True)

    # ── 3. Pre-render subtitle images (style depends on `variant`) ───────────
    subtitle_data = []   # list of (start_sec, end_sec, PIL Image, x, y)
    if words:
        font_candidates = [
            "/Library/Fonts/SF-Pro-Display-Bold.otf",
            "/Library/Fonts/SF-Pro.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font_path = next((p for p in font_candidates if os.path.exists(p)), None)
        tmp_draw  = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

        if variant == "dynamic":
            # Karaoke style: yellow active word, white for the rest, no box,
            # thick black stroke instead. Reintroduced from commit fa19e83.
            base_font_size = max(72, H // 11)
            max_text_w = int(W * 0.92)
            stroke_w   = 3
            sub_y      = int(H * 0.60)

            for ci in range(0, len(words), chunk_size):
                chunk = words[ci:ci + chunk_size]
                if not chunk:
                    continue

                chunk_texts = [wd["word"].strip().upper() for wd in chunk]
                full_text   = " ".join(chunk_texts)

                # Find the largest font size that fits the full chunk on one line
                size = base_font_size
                font = None
                while size >= 32:
                    try:
                        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
                    except Exception:
                        font = ImageFont.load_default()
                    bb = tmp_draw.textbbox((0, 0), full_text, font=font)
                    if bb[2] - bb[0] <= max_text_w:
                        break
                    size -= 4
                if font is None:
                    font = ImageFont.load_default()

                sp_bb = tmp_draw.textbbox((0, 0), " ", font=font)
                sp_w  = max(sp_bb[2] - sp_bb[0], size // 5)
                word_dims = []
                for t in chunk_texts:
                    bb = tmp_draw.textbbox((0, 0), t, font=font)
                    word_dims.append((bb[2] - bb[0], bb[3] - bb[1]))

                total_w = sum(w for w, h in word_dims) + sp_w * max(0, len(chunk_texts) - 1)
                line_h  = max((h for w, h in word_dims), default=size)
                pad     = stroke_w + 6
                img_w   = int(total_w) + pad * 2
                img_h   = int(line_h)  + pad * 2

                # One image per word -- highlight rotates through the chunk
                for wi, word in enumerate(chunk):
                    img  = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    x = pad
                    for wj, (text, (ww, wh)) in enumerate(zip(chunk_texts, word_dims)):
                        color = (255, 226, 52, 255) if wj == wi else (255, 255, 255, 255)
                        draw.text((x, pad), text, font=font, fill=color,
                                  stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
                        x += ww + sp_w
                    x_pos = max(10, (W - img_w) // 2)
                    subtitle_data.append((word["start"], word["end"], img, x_pos, sub_y))

        else:
            # Clean style: white rounded box, near-black text, sentence case.
            base_font_size = max(52, H // 16)
            max_text_w = int(W * 0.82)
            pad_h, pad_v, radius = 22, 14, 10
            text_color = (15, 15, 15, 255)
            box_color  = (255, 255, 255, 235)
            max_sub_h  = base_font_size + 2 * pad_v
            sub_y      = int(H * 0.65) - max_sub_h

            for ci in range(0, len(words), chunk_size):
                chunk = words[ci:ci + chunk_size]
                if not chunk:
                    continue

                chunk_texts = [wd["word"].strip() for wd in chunk]
                if chunk_texts:
                    chunk_texts[0] = chunk_texts[0][:1].upper() + chunk_texts[0][1:]
                full_text = " ".join(chunk_texts)

                # Find the largest font size that fits the full chunk text
                size = base_font_size
                font = None
                while size >= 28:
                    try:
                        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
                    except Exception:
                        font = ImageFont.load_default()
                    bb = tmp_draw.textbbox((0, 0), full_text, font=font)
                    if bb[2] - bb[0] <= max_text_w:
                        break
                    size -= 4
                if font is None:
                    font = ImageFont.load_default()

                bb     = tmp_draw.textbbox((0, 0), full_text, font=font)
                text_w = bb[2] - bb[0]
                text_h = bb[3] - bb[1]
                img_w  = text_w + 2 * pad_h
                img_h  = text_h + 2 * pad_v
                x_pos  = max(10, (W - img_w) // 2)

                # One subtitle image per word -- same white box, advances word by word
                for word in chunk:
                    img  = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.rounded_rectangle([(0, 0), (img_w - 1, img_h - 1)],
                                           radius=radius, fill=box_color)
                    draw.text((pad_h, pad_v - bb[1]), full_text, font=font, fill=text_color)
                    subtitle_data.append((word["start"], word["end"], img, x_pos, sub_y))

    # ── 4. Load logo ──────────────────────────────────────────────────────────
    logo_img = None
    logo_w   = max(160, W // 4)
    try:
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        lh       = int(logo_img.height * logo_w / logo_img.width)
        logo_img = logo_img.resize((logo_w, lh), Image.LANCZOS)
    except Exception:
        pass

    # ── 5. Process each frame: blur-bg composite -> overlays ─────────────────
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    for fi, fname in enumerate(frame_files):
        fp        = os.path.join(frames_dir, fname)
        frame_img = Image.open(fp).convert("RGB")
        fw, fh    = frame_img.size

        # Blurred background: scale to cover full 1080x1920 canvas, then blur
        bg_scale = max(W / fw, H / fh)
        bg = frame_img.resize((int(fw * bg_scale), int(fh * bg_scale)), Image.LANCZOS)
        bw, bh = bg.size
        bg = bg.crop(((bw - W) // 2, (bh - H) // 2, (bw - W) // 2 + W, (bh - H) // 2 + H))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=25))

        # Foreground: scale to fill 50% of canvas height, center-crop to canvas width.
        # Bars become ~25% each (down from ~34% with fit-width). Horizontal center
        # follows the tracked dominant-speaker path instead of a static midpoint,
        # so the crop follows whoever is actually on screen.
        fg_h = H // 2
        fg_w = int(fw * fg_h / fh)
        fg   = frame_img.resize((fg_w, fg_h), Image.LANCZOS)
        if fg_w > W:
            if crop_path:
                path_idx = min(fi, len(crop_path) - 1)
                src_cx   = crop_path[path_idx][0] + crop_path[path_idx][2] / 2
                cx       = int(round(src_cx * (fg_h / fh))) - W // 2
                cx       = max(0, min(cx, fg_w - W))
            else:
                cx = (fg_w - W) // 2
            fg = fg.crop((cx, 0, cx + W, fg_h))
        fg = fg.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))

        # Ken Burns zoom-in for the "dynamic" variant: ~1.0x at the first frame to
        # ~1.05x at the last, applied to the already-cropped foreground so the
        # final composited dimensions stay constant.
        if variant == "dynamic" and len(frame_files) > 1:
            t_frac = fi / (len(frame_files) - 1)
            zoom   = 1.0 + 0.05 * t_frac
            zw, zh = int(round(W * zoom)), int(round(fg_h * zoom))
            fg_zoomed = fg.resize((zw, zh), Image.LANCZOS)
            zx, zy = (zw - W) // 2, (zh - fg_h) // 2
            fg = fg_zoomed.crop((zx, zy, zx + W, zy + fg_h))

        # Center fg vertically on the blurred bg
        paste_y = (H - fg_h) // 2
        output  = bg.copy()
        output.paste(fg, (0, paste_y))

        # Convert to RGBA for alpha compositing
        output = output.convert("RGBA")

        # Paste logo (top-left area, well above subtitle zone)
        if logo_img is not None:
            output.paste(logo_img, (120, 160), logo_img)

        # Paste active subtitle
        t          = fi / fps
        active_sub = next(
            ((si, sx, sy) for s, e, si, sx, sy in subtitle_data if s <= t < e),
            None,
        )
        if active_sub is not None:
            sub_img, sx, sy = active_sub
            output.paste(sub_img, (sx, sy), sub_img)

        output.convert("RGB").save(fp)

    # ── 6. Single encode pass: frames + audio ─────────────────────────────────
    subprocess.run([
        "ffmpeg",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-ss", str(offset_seconds), "-t", str(duration), "-i", section_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-maxrate", "3500k", "-bufsize", "7000k",
        "-af", audio_filter,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p", "-shortest",
        output_path, "-y",
    ], check=True, capture_output=True)

    shutil.rmtree(frames_dir, ignore_errors=True)


# ── Subtitles ──────────────────────────────────────────────────────────────────

def words_for_clip(all_words, clip_start_s, clip_end_s):
    """Filter words that fall within the clip's time range and rebase to clip-relative seconds."""
    result = []
    for w in all_words:
        if w["end"] <= clip_start_s or w["start"] >= clip_end_s:
            continue
        result.append({
            "word": w["word"].strip(),
            "start": max(0.0, w["start"] - clip_start_s),
            "end": min(float(clip_end_s - clip_start_s), w["end"] - clip_start_s),
        })
    return result


def upload_clip(supabase, local_path, storage_path):
    """Upload to Supabase Storage public bucket, return public URL."""
    upload_url = f"{SUPABASE_URL}/storage/v1/object/clips/{storage_path}"
    with open(local_path, "rb") as f:
        data = f.read()
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "video/mp4",
        "x-upsert": "true",
    }
    for attempt in range(5):
        try:
            resp = requests.post(upload_url, data=data, headers=headers, timeout=300)
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = 15 * (attempt + 1)
            print(f"  Upload attempt {attempt+1} failed ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
    return supabase.storage.from_("clips").get_public_url(storage_path)


def queue_clip(supabase, video_id, clip_index, storage_path, public_url, caption, hook, platform):
    existing = supabase.table("clip_queue").select("id", "status").eq("video_id", video_id).eq("clip_index", clip_index).eq("platform", platform).execute()
    if existing.data:
        if existing.data[0]["status"] == "pending":
            # Update URL/caption in case clip was re-rendered
            supabase.table("clip_queue").update({
                "storage_path": storage_path,
                "public_url": public_url,
                "caption": caption,
                "hook": hook,
            }).eq("id", existing.data[0]["id"]).execute()
        return
    supabase.table("clip_queue").insert({
        "video_id": video_id,
        "clip_index": clip_index,
        "storage_path": storage_path,
        "public_url": public_url,
        "caption": caption,
        "hook": hook,
        "platform": platform,
        "status": "pending",
    }).execute()


# ── Transcript persistence ─────────────────────────────────────────────────────

def _load_transcript(supabase_admin, local_cache, video_id):
    """Load transcript from local cache or Supabase. Returns (segments, words) or (None, [])."""
    if os.path.exists(local_cache):
        with open(local_cache) as f:
            cached = json.load(f)
        if isinstance(cached, dict) and "words" in cached:
            print(f"\n[CACHED] Transcript loaded locally -- {len(cached['segments'])} segments")
            return cached["segments"], cached["words"]

    try:
        result = supabase_admin.table("video_transcripts").select("transcript").eq("video_id", video_id).execute()
        if result.data:
            cached = json.loads(result.data[0]["transcript"])
            segments, words = cached["segments"], cached.get("words", [])
            print(f"\n[SUPABASE] Transcript loaded -- {len(segments)} segments")
            with open(local_cache, "w") as f:
                json.dump(cached, f)
            return segments, words
    except Exception as e:
        print(f"  Warning: could not load transcript from Supabase: {e}")

    return None, []


def _save_transcript(supabase_admin, local_cache, video_id, segments, words):
    """Save transcript to local cache and Supabase."""
    data = {"segments": segments, "words": words}
    with open(local_cache, "w") as f:
        json.dump(data, f)
    try:
        supabase_admin.table("video_transcripts").upsert({
            "video_id": video_id,
            "transcript": json.dumps(data),
        }).execute()
        print(f"  Transcript saved to Supabase")
    except Exception as e:
        print(f"  Warning: could not save transcript to Supabase: {e}")


def _save_clip_plan(supabase_admin, video_id, clips):
    """Save Claude's clip selections to Supabase immediately after Claude responds."""
    rows = [{
        "video_id": video_id,
        "clip_index": i,
        "start_seconds": clip["start_seconds"],
        "end_seconds": clip["end_seconds"],
        "caption": json.dumps(clip.get("captions", {})),
        "hook": clip.get("hook", ""),
        "status": "pending",
    } for i, clip in enumerate(clips)]
    supabase_admin.table("video_clip_plans").upsert(rows, on_conflict="video_id,clip_index").execute()
    print(f"  Clip plan saved to Supabase ({len(rows)} clips)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max_clips", type=int, default=None, help="Limit number of clips (for testing)")
    args = parser.parse_args()

    # Retry initial Supabase connection -- DNS can flake briefly on this machine
    for _attempt in range(5):
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            supabase_admin.table("video_clip_plans").select("video_id").limit(1).execute()
            break
        except Exception as _e:
            if _attempt == 4:
                raise
            print(f"  [Network] Supabase unreachable ({_e.__class__.__name__}), retrying in 15s...")
            time.sleep(15)

    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    os.makedirs("transcripts", exist_ok=True)
    transcript_cache = f"transcripts/{args.video_id}.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        full_video_path = os.path.join(tmpdir, f"{args.video_id}_full.mp4")

        # Check for resumable clip plan in Supabase
        pending_rows = supabase_admin.table("video_clip_plans") \
            .select("*").eq("video_id", args.video_id).neq("status", "done") \
            .order("clip_index").execute()

        if pending_rows.data:
            print(f"\n[RESUME] {len(pending_rows.data)} unfinished clips found -- skipping audio/transcription/Claude")
            segments, words = _load_transcript(supabase_admin, transcript_cache, args.video_id)
            clips_to_process = []
            for row in pending_rows.data:
                try:
                    captions = json.loads(row["caption"])
                except Exception:
                    captions = {p: row["caption"] for p in ["instagram", "tiktok", "youtube", "facebook"]}
                clips_to_process.append({
                    "clip_index": row["clip_index"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "hook": row.get("hook", ""),
                    "captions": captions,
                })
            if args.max_clips:
                clips_to_process = clips_to_process[:args.max_clips]
            total_clips = supabase_admin.table("video_clip_plans") \
                .select("clip_index", count="exact").eq("video_id", args.video_id).execute()
            total = total_clips.count or len(clips_to_process)
        else:
            # PASS 1: Audio -> Transcript
            segments, words = _load_transcript(supabase_admin, transcript_cache, args.video_id)
            audio_path = None

            if segments is None:
                print(f"\n[1/2 downloads] Audio only from: {args.url}")
                audio_path = download_audio_only(args.url, tmpdir)
                size_mb = os.path.getsize(audio_path) / 1024 / 1024
                print(f"  Downloaded audio: {size_mb:.1f}MB")

                print("[Transcribing]")
                segments, words = transcribe(audio_path)
                print(f"  {len(segments)} transcript segments, {len(words)} words")
                _save_transcript(supabase_admin, transcript_cache, args.video_id, segments, words)

            # Full video is downloaded early (rather than only after selection, as
            # before) so local visual/audio signals can be computed before Claude
            # picks clips -- same total download bytes, just reordered.
            print(f"\n[2/2 downloads] Downloading full video at 720p (this takes a few minutes)...")
            download_full_video(args.url, full_video_path)
            full_mb = os.path.getsize(full_video_path) / 1024 / 1024
            print(f"  Downloaded: {full_mb:.0f}MB")

            print("\n[Signals] Computing visual/audio energy signals...")
            try:
                signaled_segments = _compute_segment_signals(
                    full_video_path, segments, audio_source=audio_path or full_video_path,
                )
                n_tagged = sum(1 for s in signaled_segments if s.get("signal_tag"))
                print(f"  Tagged {n_tagged}/{len(signaled_segments)} segments.")
            except Exception as e:
                print(f"  [Signals] failed ({e}) -- selecting from transcript only")
                signaled_segments = segments

            # Claude picks clips
            print("\n[Claude] Selecting viral clips from transcript...")
            all_clips = select_clips(anthropic_client, signaled_segments, args.title, supabase_admin)
            print(f"  {len(all_clips)} clips selected")

            if not all_clips:
                print("No valid clips returned by Claude. Exiting.", file=sys.stderr)
                sys.exit(1)

            if args.max_clips:
                all_clips = all_clips[:args.max_clips]

            _save_clip_plan(supabase_admin, args.video_id, all_clips)
            _log_clip_selections(supabase_admin, args.video_id, all_clips, segments=segments)

            for i, clip in enumerate(all_clips):
                print(f"  [{i+1}] {to_hhmmss(clip['start_seconds'])} -> {to_hhmmss(clip['end_seconds'])} | {clip['hook'][:65]}")

            clips_to_process = [{"clip_index": i, **clip} for i, clip in enumerate(all_clips)]
            total = len(all_clips)

        # PASS 2: cut each clip locally from the full video.
        # In the fresh-processing branch above, the full video was already
        # downloaded early (to compute signals before selection); only the
        # RESUME branch (crash recovery, skips straight here) still needs it now.
        if not os.path.exists(full_video_path):
            print(f"\n[2/2 downloads] Downloading full video at 720p (this takes a few minutes)...")
            download_full_video(args.url, full_video_path)
        full_mb = os.path.getsize(full_video_path) / 1024 / 1024
        print(f"  Full video: {full_mb:.0f}MB -> cutting {len(clips_to_process)} clips locally...")

        succeeded = 0
        for item in clips_to_process:
            i = item["clip_index"]
            start_s = item["start_seconds"]
            end_s = item["end_seconds"]
            duration = end_s - start_s

            clip_path = os.path.join(tmpdir, f"{args.video_id}_clip_{i}.mp4")
            storage_path = f"{args.video_id}/{args.video_id}_clip_{i}.mp4"

            print(f"\n  [{i+1}/{total}] {to_hhmmss(start_s)} -> {to_hhmmss(end_s)} ({duration:.0f}s)")
            try:
                clip_words = words_for_clip(words, start_s, end_s) if words else []
                clip_variant = _clip_variant(args.video_id, i)
                print(f"  Cutting, cropping, burning subtitles ({len(clip_words)} words, variant={clip_variant})...")
                clip_hook = item.get("hook", "")
                try:
                    cut_and_subtitle(full_video_path, start_s, duration, clip_words, clip_path, i, tmpdir, hook=clip_hook, variant=clip_variant)
                except Exception as e:
                    print(f"  cut_and_subtitle failed ({e}), retrying without subtitles...")
                    cut_and_subtitle(full_video_path, start_s, duration, [], clip_path, i, tmpdir, hook=clip_hook, variant=clip_variant)

                clip_mb = os.path.getsize(clip_path) / 1024 / 1024
                print(f"  Uploading ({clip_mb:.1f}MB)...")
                public_url = upload_clip(supabase_admin, clip_path, storage_path)

                captions = item.get("captions", {})
                for platform in ["instagram", "tiktok", "youtube", "facebook"]:
                    caption = captions.get(platform) or item.get("caption", "")
                    queue_clip(supabase_admin, args.video_id, i, storage_path, public_url, caption, item.get("hook", ""), platform)

                print(f"  Queued: {public_url[:60]}...")
                supabase_admin.table("video_clip_plans").update({"status": "done"}) \
                    .eq("video_id", args.video_id).eq("clip_index", i).execute()
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                succeeded += 1

            except Exception as e:
                print(f"  x Clip {i+1} failed: {e} -- skipping, continuing...")
                supabase_admin.table("video_clip_plans").update({"status": "failed"}) \
                    .eq("video_id", args.video_id).eq("clip_index", i).execute()
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                continue

        # Mark processed
        if succeeded > 0:
            existing = supabase_admin.table("processed_videos").select("id") \
                .eq("video_id", args.video_id).execute()
            if not existing.data:
                supabase_admin.table("processed_videos").insert({
                    "video_id": args.video_id,
                    "video_title": args.title,
                    "channel_id": CHANNEL_ID,
                    "clip_count": succeeded,
                }).execute()

            platforms = ["instagram", "tiktok", "youtube", "facebook"]
            print(f"\nDone. {succeeded}/{total} clips queued -- {succeeded * len(platforms)} posts scheduled.")
            if succeeded < total:
                print(f"  {total - succeeded} clips failed. Re-run to retry -- clip plan is saved in Supabase.")
        else:
            print(f"\nAll {total} clips failed.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
