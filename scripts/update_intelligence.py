#!/usr/bin/env python3
"""
Analyses historical clip performance and extracts actionable patterns
that the AI uses when selecting clips for future videos.

Runs daily via GitHub Actions. Reads analytics + Supabase clip metadata,
researches current platform algorithms via web search, and saves the
resulting intelligence brief back to Supabase so process_video.py can
inject it into the clip-selection prompt.
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ZERNIO_KEY_1      = os.environ["ZERNIO_API_KEY"]    # instagram + tiktok
ZERNIO_KEY_2      = os.environ["ZERNIO_API_KEY_2"]  # youtube + facebook

import anthropic
from supabase import create_client

sb     = create_client(SUPABASE_URL, SUPABASE_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Stable anchor repeated in every weekly brief -- not just prose scene-setting,
# the actual thing "is the current strategy working?" gets checked against.
GOAL = (
    "Grow real reach and followers for the Konnected Minds Podcast (Ghana/Africa "
    "entrepreneurship audience) across TikTok, Instagram Reels, YouTube Shorts, and "
    "Facebook Reels -- not just rack up per-clip views, but build a returning audience."
)


# ── Zernio helpers ─────────────────────────────────────────────────────────────

def fetch_zernio_all_posts() -> dict[str, dict]:
    """Fetch all posts from both accounts. Returns latePostId → post dict."""
    posts_map: dict[str, dict] = {}
    for key in [ZERNIO_KEY_1, ZERNIO_KEY_2]:
        page = 1
        while True:
            resp = requests.get(
                f"https://zernio.com/api/v1/analytics?page={page}",
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
            if not resp.ok:
                break
            data = resp.json()
            for post in data.get("posts", []):
                if post.get("latePostId"):
                    posts_map[post["latePostId"]] = post
            pagination = data.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break
            page += 1
    return posts_map


def fetch_follower_counts() -> dict[str, int]:
    """
    Probe Zernio account endpoints for follower counts per platform.
    Returns a dict like {"instagram": 1420, "tiktok": 830, ...} or empty if unavailable.
    """
    follower_counts: dict[str, int] = {}
    PLATFORM_KEYS = {
        "instagram": ZERNIO_KEY_1,
        "tiktok":    ZERNIO_KEY_1,
        "youtube":   ZERNIO_KEY_2,
        "facebook":  ZERNIO_KEY_2,
    }
    probe_endpoints = ["/api/v1/accounts", "/api/v1/channels", "/api/v1/profile", "/api/v1/stats"]

    for key in set(PLATFORM_KEYS.values()):
        for endpoint in probe_endpoints:
            try:
                resp = requests.get(
                    f"https://zernio.com{endpoint}",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10,
                )
                if not resp.ok:
                    continue
                data = resp.json()
                raw = json.dumps(data).lower()
                print(f"  Zernio {endpoint}: {raw[:300]}")
                # Parse follower/subscriber counts from whatever structure comes back
                for item in (data if isinstance(data, list) else [data]):
                    platform = (
                        item.get("platform") or item.get("network") or item.get("type") or ""
                    ).lower()
                    for field in ["followers", "followerCount", "subscribers", "subscriberCount", "fans"]:
                        count = item.get(field) or item.get(field.lower())
                        if count and isinstance(count, int) and platform:
                            follower_counts[platform] = count
                if follower_counts:
                    break
            except Exception:
                continue
        if follower_counts:
            break

    if not follower_counts:
        print("  Follower counts not available via API (endpoint not found).")
    return follower_counts


# ── Supabase helpers ───────────────────────────────────────────────────────────

def fetch_clip_metadata() -> list[dict]:
    """Pull posted clips with hooks, captions, durations, and timestamps."""
    clips = (
        sb.table("clip_queue")
        .select("video_id, clip_index, platform, zernio_post_id, hook, caption, posted_at")
        .eq("status", "posted")
        .not_.is_("zernio_post_id", "null")
        .not_.is_("hook", "null")
        .order("posted_at", desc=True)
        .limit(500)
        .execute()
    ).data or []

    plans_raw = (
        sb.table("video_clip_plans")
        .select("video_id, clip_index, start_seconds, end_seconds")
        .execute()
    ).data or []
    plans = {(p["video_id"], p["clip_index"]): p for p in plans_raw}

    for clip in clips:
        key = (clip["video_id"], clip["clip_index"])
        plan = plans.get(key)
        clip["duration_seconds"] = (
            round(plan["end_seconds"] - plan["start_seconds"]) if plan else None
        )
        try:
            clip["captions"] = json.loads(clip["caption"]) if isinstance(clip["caption"], str) else (clip["caption"] or {})
        except Exception:
            clip["captions"] = {}

    return clips


def save_intelligence(summary: str, stats: dict) -> None:
    sb.table("channel_intelligence").upsert(
        {
            "id":         "singleton",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "summary":    summary,
            "stats":      stats,
        },
        on_conflict="id",
    ).execute()
    print("  Saved channel intelligence to Supabase.")

    # Insert-only audit trail so the brief/recommendations' evolution over time
    # is visible, not just the latest snapshot (which the line above overwrites).
    try:
        sb.table("intelligence_history").insert({
            "summary": summary,
            "stats":   stats,
        }).execute()
        print("  Logged this review to intelligence_history.")
    except Exception as e:
        print(f"  Warning: could not log to intelligence_history ({e}) -- "
              f"run scripts/migrate_add_intelligence_history.py if not done yet.")


# ── YouTube Analytics ──────────────────────────────────────────────────────────

def fetch_youtube_analytics() -> str:
    """Pull YouTube Analytics data for the last 30 days using OAuth credentials."""
    client_id     = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        print("  YouTube Analytics: credentials not configured, skipping.")
        return ""

    try:
        # Refresh access token
        token_resp = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id":     client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type":    "refresh_token",
        })
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=30)).strftime("%Y-%m-%d")

        base_params = {
            "ids":       "channel==MINE",
            "startDate": start_date,
            "endDate":   end_date,
        }

        # Overall channel metrics
        channel_resp = requests.get(
            "https://youtubeanalytics.googleapis.com/v2/reports",
            headers=headers,
            params={**base_params, "metrics": "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost"},
        )
        channel_resp.raise_for_status()
        channel_data = channel_resp.json()

        # Traffic sources
        traffic_resp = requests.get(
            "https://youtubeanalytics.googleapis.com/v2/reports",
            headers=headers,
            params={**base_params, "metrics": "views", "dimensions": "insightTrafficSourceType", "sort": "-views"},
        )
        traffic_resp.raise_for_status()
        traffic_data = traffic_resp.json()

        # Top videos by views
        videos_resp = requests.get(
            "https://youtubeanalytics.googleapis.com/v2/reports",
            headers=headers,
            params={**base_params, "metrics": "views,averageViewPercentage,averageViewDuration,likes,shares,comments", "dimensions": "video", "sort": "-views", "maxResults": 10},
        )
        videos_resp.raise_for_status()
        videos_data = videos_resp.json()

        # Format output
        lines = ["YOUTUBE ANALYTICS (last 30 days):"]

        if channel_data.get("rows"):
            row = channel_data["rows"][0]
            cols = [h["name"] for h in channel_data["columnHeaders"]]
            d = dict(zip(cols, row))
            lines.append(f"  Views: {int(d.get('views', 0)):,}")
            lines.append(f"  Avg view duration: {int(d.get('averageViewDuration', 0))}s")
            lines.append(f"  Avg view percentage: {round(d.get('averageViewPercentage', 0), 1)}%")
            lines.append(f"  Subscribers gained: {int(d.get('subscribersGained', 0))}")
            lines.append(f"  Subscribers lost: {int(d.get('subscribersLost', 0))}")

        if traffic_data.get("rows"):
            lines.append("  Traffic sources:")
            for row in traffic_data["rows"][:6]:
                source, views = row[0], int(row[1])
                readable = {
                    "YT_SEARCH": "YouTube Search",
                    "SUBSCRIBER": "Subscribers",
                    "SHORTS": "Shorts shelf",
                    "SUGGESTED_VIDEOS": "Suggested videos",
                    "NO_LINK_OTHER": "Direct/other",
                    "EXTERNAL_APP": "External apps",
                    "NOTIFICATION": "Notifications",
                }.get(source, source)
                lines.append(f"    {readable}: {views:,} views")

        if videos_data.get("rows"):
            cols = [h["name"] for h in videos_data["columnHeaders"]]
            lines.append("  Top 10 Shorts (by views):")
            for row in videos_data["rows"]:
                d = dict(zip(cols, row))
                lines.append(
                    f"    video/{d['video']}: {int(d.get('views',0)):,} views | "
                    f"{round(d.get('averageViewPercentage',0),1)}% retention | "
                    f"{int(d.get('averageViewDuration',0))}s avg watch | "
                    f"{int(d.get('likes',0))} likes | {int(d.get('shares',0))} shares"
                )

        print("  YouTube Analytics: data fetched successfully.")
        return "\n".join(lines)

    except Exception as e:
        print(f"  YouTube Analytics fetch failed: {e}")
        return ""


# ── Algorithm research ─────────────────────────────────────────────────────────

def research_platform_algorithms() -> str:
    """
    Research current platform algorithm priorities via web search.
    Cached in Supabase settings for 7 days to limit cost.
    Falls back to Claude's training knowledge if web search is unavailable.
    """
    CACHE_KEY = "algorithm_research_v2"

    try:
        cached = sb.table("settings").select("value").eq("key", CACHE_KEY).single().execute()
        if cached.data:
            val = cached.data.get("value") or {}
            updated_at = val.get("updated_at", "")
            if updated_at:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).days
                if age < 3 and val.get("text"):
                    print(f"  Using cached algorithm research ({age}d old).")
                    return val["text"]
    except Exception:
        pass

    print("  Running platform algorithm research...")

    prompt = """You are a social media growth expert researching the current state of short-form video algorithms in 2025.

Research and write specific, actionable insights on the algorithm priorities for each platform:

TIKTOK: What signals the algorithm prioritises right now (completion rate, shares, saves, early engagement velocity, replays, etc.), what content types it is currently boosting vs suppressing, how it distributes content to non-followers, what drives real follower growth vs just views.

INSTAGRAM REELS: How the algorithm distributes to non-followers, what actions it rewards most (saves, shares, comments, DMs), what the Explore page prioritises, how non-follower reach works in 2025.

YOUTUBE SHORTS: What drives recommendations and the Shorts shelf, how video titles and thumbnails affect CTR, what watch signals matter, what drives subscriber conversion.

FACEBOOK REELS: Who it reaches and how, what engagement signals matter most, how it differs from Instagram, how to use it for actual audience growth.

For each platform: focus on what ACTUALLY drives follower growth and long-term audience building, not just one-off viral views. Include any significant algorithm changes in the last 12 months worth knowing.

Context: Konnected Minds is an African entrepreneurship podcast channel (Ghana-based) posting 60-second clips 3 times per day. Audience is business-minded 20-45 year olds interested in money, hustle, real stories, and Africa-specific business topics.

Write specific, data-backed insights only. No padding."""

    text = ""

    # Try with web search tool first
    try:
        messages = [{"role": "user", "content": prompt}]
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )
        for _ in range(6):
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in response.content
                if hasattr(b, "type") and b.type == "tool_use"
            ]
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages,
            )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        if text:
            print("  Web search algorithm research complete.")
    except Exception as e:
        print(f"  Web search unavailable ({type(e).__name__}), using knowledge base.")

    # Fallback: Claude's training knowledge
    if not text:
        try:
            resp = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            print("  Knowledge-based algorithm research complete.")
        except Exception as e:
            print(f"  Algorithm research failed: {e}")
            return ""

    # Cache result
    try:
        sb.table("settings").upsert({
            "key":   CACHE_KEY,
            "value": {"text": text, "updated_at": datetime.now(timezone.utc).isoformat()},
        }, on_conflict="key").execute()
    except Exception:
        pass

    return text


# ── Decision outcome tracking ──────────────────────────────────────────────────

def update_selection_outcomes(zernio_map: dict[str, dict]) -> None:
    """
    Match logged clip selections to real analytics and fill in performance data.
    Assigns a performance tier (top / mid / low) relative to channel average.
    """
    try:
        pending = (
            sb.table("clip_selection_log")
            .select("id, video_id, clip_index")
            .is_("analytics_updated_at", "null")
            .execute()
        ).data or []

        if not pending:
            print("  No pending selection outcomes to update.")
            return

        queue_rows = (
            sb.table("clip_queue")
            .select("video_id, clip_index, platform, zernio_post_id")
            .eq("status", "posted")
            .not_.is_("zernio_post_id", "null")
            .execute()
        ).data or []

        best_analytics: dict[tuple, dict] = {}
        for row in queue_rows:
            key = (row["video_id"], row["clip_index"])
            zpost = zernio_map.get(row["zernio_post_id"] or "")
            analytics = (zpost or {}).get("analytics") or {}
            if not analytics.get("lastUpdated"):
                continue
            prev = best_analytics.get(key, {})
            if (analytics.get("views") or 0) >= (prev.get("views") or 0):
                best_analytics[key] = analytics

        all_views = [a.get("views") or 0 for a in best_analytics.values() if a.get("views")]
        avg_views = sum(all_views) / len(all_views) if all_views else 0

        updated = 0
        for rec in pending:
            key = (rec["video_id"], rec["clip_index"])
            a = best_analytics.get(key)
            if not a:
                continue
            views = a.get("views") or 0
            tier = "top" if views >= avg_views * 1.5 else ("low" if views < avg_views * 0.5 else "mid")
            sb.table("clip_selection_log").update({
                "views":                views,
                "likes":                a.get("likes") or 0,
                "engagement_rate":      round(a.get("engagementRate") or 0, 2),
                "performance_tier":     tier,
                "analytics_updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", rec["id"]).execute()
            updated += 1

        print(f"  Updated outcomes for {updated} clip selections.")
    except Exception as e:
        print(f"  Could not update selection outcomes: {e}")


def build_decision_history(zernio_map: dict[str, dict]) -> str:
    """
    Build a summary of past clip selection decisions and their outcomes,
    broken down cross-platform and per-platform, for injection into the brief.
    """
    try:
        rows = (
            sb.table("clip_selection_log")
            .select("video_id, clip_index, hook_type, topic_category, duration_seconds, views, performance_tier")
            .not_.is_("performance_tier", "null")
            .execute()
        ).data or []

        if len(rows) < 10:
            return ""

        queue_rows = (
            sb.table("clip_queue")
            .select("video_id, clip_index, platform, zernio_post_id")
            .eq("status", "posted")
            .not_.is_("zernio_post_id", "null")
            .execute()
        ).data or []

        queue_by_clip: dict[tuple, list] = defaultdict(list)
        for qrow in queue_rows:
            key = (qrow["video_id"], qrow["clip_index"])
            zpost = zernio_map.get(qrow.get("zernio_post_id") or "")
            analytics = (zpost or {}).get("analytics") or {}
            if analytics.get("views"):
                queue_by_clip[key].append({
                    "platform":        qrow["platform"],
                    "views":           analytics.get("views") or 0,
                    "engagement_rate": round(analytics.get("engagementRate") or 0, 2),
                })

        hook_stats:  dict[str, list] = defaultdict(list)
        topic_stats: dict[str, list] = defaultdict(list)
        tier_counts: dict[str, int]  = defaultdict(int)
        platform_hook:  dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        platform_topic: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for r in rows:
            if r.get("views") is not None:
                hook_stats[r["hook_type"] or "unknown"].append(r["views"])
                topic_stats[r["topic_category"] or "unknown"].append(r["views"])
            tier_counts[r["performance_tier"] or "unknown"] += 1

            key = (r["video_id"], r["clip_index"])
            for pdata in queue_by_clip.get(key, []):
                platform_hook[pdata["platform"]][r["hook_type"] or "unknown"].append(pdata["views"])
                platform_topic[pdata["platform"]][r["topic_category"] or "unknown"].append(pdata["views"])

        lines = [f"PAST CLIP SELECTION OUTCOMES ({len(rows)} clips with data):"]

        lines.append("\nBy hook type (avg views, cross-platform):")
        for ht, vl in sorted(hook_stats.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            lines.append(f"  {ht}: {round(sum(vl)/len(vl))} avg views ({len(vl)} clips)")

        lines.append("\nBy topic (avg views, cross-platform):")
        for tc, vl in sorted(topic_stats.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            lines.append(f"  {tc}: {round(sum(vl)/len(vl))} avg views ({len(vl)} clips)")

        if platform_hook:
            lines.append("\nPER-PLATFORM HOOK PERFORMANCE:")
            for platform in ["tiktok", "instagram", "youtube", "facebook"]:
                if platform not in platform_hook:
                    continue
                lines.append(f"\n  {platform.upper()}:")
                for ht, vl in sorted(platform_hook[platform].items(), key=lambda x: -sum(x[1]) / len(x[1]))[:5]:
                    lines.append(f"    {ht}: {round(sum(vl)/len(vl))} avg views ({len(vl)} clips)")

        if platform_topic:
            lines.append("\nPER-PLATFORM TOPIC PERFORMANCE:")
            for platform in ["tiktok", "instagram", "youtube", "facebook"]:
                if platform not in platform_topic:
                    continue
                lines.append(f"\n  {platform.upper()}:")
                for tc, vl in sorted(platform_topic[platform].items(), key=lambda x: -sum(x[1]) / len(x[1]))[:5]:
                    lines.append(f"    {tc}: {round(sum(vl)/len(vl))} avg views ({len(vl)} clips)")

        total = sum(tier_counts.values())
        if total > 0:
            lines.append(
                f"\nOverall: {tier_counts.get('top', 0)} top "
                f"({round(tier_counts.get('top', 0) / total * 100)}%), "
                f"{tier_counts.get('mid', 0)} mid, {tier_counts.get('low', 0)} low performers"
            )

        return "\n".join(lines)
    except Exception as e:
        print(f"  Warning: could not build decision history: {e}")
        return ""


def build_variant_analysis(zernio_map: dict[str, dict]) -> str:
    """
    Correlate caption_variant ("clean" vs "dynamic", see cut_and_subtitle in
    process_video.py) with real view performance -- the A/B test set up in the
    presentation phase. Returns "" until enough posted+synced data exists on
    both sides to say anything meaningful.
    """
    try:
        rows = (
            sb.table("clip_selection_log")
            .select("video_id, clip_index, caption_variant")
            .not_.is_("caption_variant", "null")
            .execute()
        ).data or []

        if len(rows) < 10:
            return ""

        queue_rows = (
            sb.table("clip_queue")
            .select("video_id, clip_index, platform, zernio_post_id")
            .eq("status", "posted")
            .not_.is_("zernio_post_id", "null")
            .execute()
        ).data or []

        queue_by_clip: dict[tuple, list] = defaultdict(list)
        for qrow in queue_rows:
            key = (qrow["video_id"], qrow["clip_index"])
            zpost = zernio_map.get(qrow.get("zernio_post_id") or "")
            analytics = (zpost or {}).get("analytics") or {}
            if analytics.get("views"):
                queue_by_clip[key].append({
                    "platform": qrow["platform"],
                    "views":    analytics.get("views") or 0,
                })

        variant_stats:    dict[str, list] = defaultdict(list)
        platform_variant: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for r in rows:
            key = (r["video_id"], r["clip_index"])
            variant = r["caption_variant"]
            for pdata in queue_by_clip.get(key, []):
                variant_stats[variant].append(pdata["views"])
                platform_variant[pdata["platform"]][variant].append(pdata["views"])

        if len(variant_stats) < 2:
            return ""

        lines = [f"CAPTION STYLE A/B RESULTS ({len(rows)} clips tagged, clean vs dynamic):"]
        for variant, vl in sorted(variant_stats.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            lines.append(f"  {variant}: {round(sum(vl)/len(vl))} avg views ({len(vl)} synced posts)")

        if platform_variant:
            lines.append("\nPER-PLATFORM:")
            for platform in ["tiktok", "instagram", "youtube", "facebook"]:
                if platform not in platform_variant or len(platform_variant[platform]) < 2:
                    continue
                lines.append(f"\n  {platform.upper()}:")
                for variant, vl in sorted(platform_variant[platform].items(), key=lambda x: -sum(x[1]) / len(x[1])):
                    lines.append(f"    {variant}: {round(sum(vl)/len(vl))} avg views ({len(vl)} posts)")

        return "\n".join(lines)
    except Exception as e:
        print(f"  Warning: could not build variant analysis: {e}")
        return ""


# ── Content pattern analysis ───────────────────────────────────────────────────

def build_content_analysis(zernio_map: dict[str, dict]) -> str:
    """
    Compare the actual transcript text of top vs bottom performing clips.
    Identifies structural patterns — openings, endings, content density, arc —
    that separate clips that perform well from those that don't.
    """
    try:
        # Prefer clips with transcript text stored
        rows = (
            sb.table("clip_selection_log")
            .select("hook, clip_transcript, hook_type, topic_category, duration_seconds, performance_tier, views")
            .not_.is_("performance_tier", "null")
            .execute()
        ).data or []

        has_transcripts = any(r.get("clip_transcript") for r in rows)
        top_clips    = [r for r in rows if r.get("performance_tier") == "top"]
        bottom_clips = [r for r in rows if r.get("performance_tier") == "low"]

        if len(top_clips) < 3:
            return ""

        if has_transcripts:
            def fmt(r: dict) -> str:
                parts = [f'Hook: "{r.get("hook", "")}"']
                if r.get("clip_transcript"):
                    parts.append(f"Full text: {r['clip_transcript'][:500]}")
                if r.get("duration_seconds"):
                    parts.append(f"Duration: {r['duration_seconds']}s")
                return "\n  ".join(parts)

            top_block    = "\n\n".join(fmt(r) for r in top_clips[:8])
            bottom_block = "\n\n".join(fmt(r) for r in bottom_clips[:6]) if bottom_clips else "Not enough low-performer data yet."

            prompt = f"""You are a clip-cutting specialist studying what makes short-form video clips succeed or fail.

Analyse the FULL SPOKEN CONTENT of these clips and identify what structurally separates top performers from low performers.

TOP PERFORMING CLIPS (high views):
{top_block}

LOW PERFORMING CLIPS (very few views):
{bottom_block}

Write specific, actionable rules for cutting better clips, covering:
1. OPENING — How do the best clips begin? What is happening in the first 3 seconds? How do weak openers fail?
2. STRUCTURE — Do top clips have a 3-part arc (setup, development, payoff)? Do low clips ramble or trail off?
3. ENDING — How do the best clips END? Do they land on a punchline, a strong statement, or a specific result? What does a weak ending sound like?
4. CONTENT DENSITY — One tight idea vs multiple scattered points. What does the data show?
5. SPECIFICITY — Do the best clips have names, numbers, places, amounts? Or are they vague?

Write 6-8 concrete rules the clip-selection AI should follow when choosing WHERE to cut a clip and what content to include. Each rule should be immediately applicable."""

        else:
            # No transcript yet — analyse hook text patterns only
            top_hooks    = [f'"{r["hook"]}" ({r.get("duration_seconds") or "?"}s)' for r in top_clips[:10]]
            bottom_hooks = [f'"{r["hook"]}" ({r.get("duration_seconds") or "?"}s)' for r in bottom_clips[:8]]

            prompt = f"""You are a clip-cutting specialist studying what makes short-form video hooks succeed or fail.

TOP PERFORMING HOOKS (high views):
{chr(10).join(f'{i+1}. {h}' for i, h in enumerate(top_hooks))}

LOW PERFORMING HOOKS (very few views):
{chr(10).join(f'{i+1}. {h}' for i, h in enumerate(bottom_hooks)) if bottom_hooks else "Not enough data yet."}

Based on these hooks, write 5-7 specific, concrete rules for choosing better clip opening moments. Focus on:
- What structural elements make the top hooks immediately grab attention?
- What makes the weak hooks fail?
- What is the single most important difference between the two groups?
Reference specific examples from the data above."""

        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        prefix = "CLIP CONTENT ANALYSIS (full transcript)" if has_transcripts else "CLIP CONTENT ANALYSIS (from hook text)"
        return f"{prefix}:\n{resp.content[0].text}"
    except Exception as e:
        print(f"  Warning: content analysis failed: {e}")
        return ""


# ── Analysis ───────────────────────────────────────────────────────────────────

def build_performance_dataset(clips: list[dict], zernio_map: dict[str, dict]) -> list[dict]:
    """
    Merge clip metadata with live analytics.
    Includes timing, caption length, and engagement breakdown per post.
    """
    rows = []
    for clip in clips:
        zpost = zernio_map.get(clip["zernio_post_id"])
        if not zpost:
            continue
        analytics     = zpost.get("analytics") or {}
        views         = analytics.get("views") or 0
        likes         = analytics.get("likes") or 0
        comments      = analytics.get("comments") or 0
        shares        = analytics.get("shares") or 0
        eng_rate      = analytics.get("engagementRate") or 0
        last_updated  = analytics.get("lastUpdated")

        if not last_updated or views == 0:
            continue

        # Parse timing from posted_at
        posted_at = clip.get("posted_at") or ""
        day_of_week = ""
        hour_utc    = None
        try:
            dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            day_of_week = dt.strftime("%A")
            hour_utc    = dt.hour
        except Exception:
            pass

        # Caption length for this platform
        captions = clip.get("captions") or {}
        platform_caption = captions.get(clip["platform"]) or ""
        caption_length = len(platform_caption.strip())

        rows.append({
            "platform":         clip["platform"],
            "hook":             (clip.get("hook") or "")[:120],
            "duration_seconds": clip.get("duration_seconds"),
            "views":            views,
            "likes":            likes,
            "comments":         comments,
            "shares":           shares,
            "engagement_rate":  round(eng_rate, 2),
            "posted_at":        posted_at[:10],
            "day_of_week":      day_of_week,
            "hour_utc":         hour_utc,
            "caption_length":   caption_length,
        })

    return sorted(rows, key=lambda r: r["views"], reverse=True)


def build_timing_analysis(rows: list[dict]) -> str:
    """Analyse which days of the week and posting times drive the most views."""
    if not rows:
        return ""

    day_views:  dict[str, list] = defaultdict(list)
    hour_views: dict[int, list]  = defaultdict(list)
    platform_hour: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        if r.get("day_of_week"):
            day_views[r["day_of_week"]].append(r["views"])
        if r.get("hour_utc") is not None:
            hour_views[r["hour_utc"]].append(r["views"])
            platform_hour[r["platform"]][r["hour_utc"]].append(r["views"])

    if not day_views:
        return ""

    lines = ["TIMING PERFORMANCE:"]

    days_ranked = sorted(day_views.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    lines.append("\nBest days to post (avg views):")
    for day, vl in days_ranked:
        lines.append(f"  {day}: {round(sum(vl)/len(vl))} avg views ({len(vl)} posts)")

    if hour_views:
        lines.append("\nBest posting times UTC (top 4):")
        hours_ranked = sorted(hour_views.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:4]
        for h, vl in hours_ranked:
            lines.append(f"  {h:02d}:00 UTC: {round(sum(vl)/len(vl))} avg views ({len(vl)} posts)")

    for platform in ["tiktok", "instagram", "youtube", "facebook"]:
        if platform not in platform_hour:
            continue
        ph = platform_hour[platform]
        if not ph:
            continue
        best_hour = max(ph.items(), key=lambda x: sum(x[1]) / len(x[1]))
        avg = round(sum(best_hour[1]) / len(best_hour[1]))
        lines.append(f"  {platform.upper()} best hour: {best_hour[0]:02d}:00 UTC ({avg} avg views)")

    return "\n".join(lines)


def build_caption_analysis(rows: list[dict]) -> str:
    """Analyse caption length vs engagement per platform."""
    if not rows:
        return ""

    SHORT  = 80   # chars
    MEDIUM = 220

    platform_buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"short": [], "medium": [], "long": []})

    for r in rows:
        length = r.get("caption_length") or 0
        if length == 0:
            continue
        bucket = "short" if length < SHORT else ("medium" if length < MEDIUM else "long")
        platform_buckets[r["platform"]][bucket].append(r["views"])

    lines = ["CAPTION LENGTH VS VIEWS (per platform):"]
    for platform in ["tiktok", "instagram", "youtube", "facebook"]:
        if platform not in platform_buckets:
            continue
        pb = platform_buckets[platform]
        line_parts = []
        for bucket in ["short", "medium", "long"]:
            vl = pb[bucket]
            if vl:
                label = f"short (<{SHORT}c)" if bucket == "short" else (f"medium ({SHORT}-{MEDIUM}c)" if bucket == "medium" else f"long (>{MEDIUM}c)")
                line_parts.append(f"{label}: {round(sum(vl)/len(vl))} avg views ({len(vl)} posts)")
        if line_parts:
            lines.append(f"\n  {platform.upper()}: " + " | ".join(line_parts))

    return "\n".join(lines) if len(lines) > 1 else ""


def build_goal_progress(rows: list[dict]) -> str:
    """
    Week-over-week comparison so the brief can state whether GOAL is actually
    being met right now -- growing or flat/declining -- not just describe
    all-time patterns. `rows` is build_performance_dataset()'s output, which
    already carries posted_at (YYYY-MM-DD), platform, views, engagement_rate.
    """
    if not rows:
        return ""

    today            = datetime.now(timezone.utc).date()
    this_week_start  = today - timedelta(days=7)
    last_week_start  = today - timedelta(days=14)

    def in_range(date_str, start, end):
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            return start <= d < end
        except Exception:
            return False

    this_week = [r for r in rows if in_range(r.get("posted_at", ""), this_week_start, today + timedelta(days=1))]
    last_week = [r for r in rows if in_range(r.get("posted_at", ""), last_week_start, this_week_start)]

    if not this_week and not last_week:
        return ""

    def platform_avgs(subset):
        by_platform: dict[str, list] = defaultdict(list)
        for r in subset:
            by_platform[r["platform"]].append(r)
        out = {}
        for p, rs in by_platform.items():
            views = [r["views"] for r in rs]
            out[p] = {"avg_views": sum(views) / len(views) if views else 0, "count": len(rs)}
        return out

    this = platform_avgs(this_week)
    last = platform_avgs(last_week)

    lines = [f"GOAL PROGRESS (this week vs last week, {len(this_week)} vs {len(last_week)} synced posts):"]
    for platform in ["tiktok", "instagram", "youtube", "facebook"]:
        t, l = this.get(platform), last.get(platform)
        if not t and not l:
            continue
        if t and l and l["avg_views"] > 0:
            delta_pct = round((t["avg_views"] - l["avg_views"]) / l["avg_views"] * 100)
            sign = "+" if delta_pct >= 0 else ""
            lines.append(
                f"  {platform.upper()}: {round(t['avg_views'])} avg views this week vs "
                f"{round(l['avg_views'])} avg views last week ({sign}{delta_pct}%), {t['count']} posts this week"
            )
        elif t:
            lines.append(f"  {platform.upper()}: {round(t['avg_views'])} avg views this week ({t['count']} posts) -- no synced data last week to compare yet")
        elif l:
            lines.append(f"  {platform.upper()}: no synced posts this week yet (last week: {round(l['avg_views'])} avg views, {l['count']} posts)")

    return "\n".join(lines) if len(lines) > 1 else ""


def build_growth_analysis() -> str:
    """
    Reads clip_performance time-series snapshots to find growth patterns.
    Shows how views accumulate over days for different hook types and platforms.
    Reveals which clips have slow burns vs early spikes, and which are still growing.
    """
    try:
        rows = (
            sb.table("clip_performance")
            .select("platform, hook, hook_type, views, likes, shares, saves, hours_since_posted, measured_at")
            .order("hours_since_posted")
            .limit(2000)
            .execute()
        ).data or []

        if len(rows) < 10:
            return ""

        # Group snapshots by clip identity (platform + hook truncated as proxy key)
        # Build: for each hook_type, what are avg views at 24h, 72h, 168h (7d)?
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            ht = r.get("hook_type") or "unknown"
            h  = r.get("hours_since_posted") or 0
            v  = r.get("views") or 0
            bucket = "24h" if h <= 24 else "72h" if h <= 72 else "7d" if h <= 168 else "30d"
            by_type[ht][bucket].append(v)

        lines = ["VIEW GROWTH BY HOOK TYPE (from daily time-series snapshots):"]
        for ht, buckets in sorted(by_type.items()):
            parts = []
            for label in ["24h", "72h", "7d", "30d"]:
                vals = buckets.get(label, [])
                if vals:
                    parts.append(f"{label}: avg {int(sum(vals)/len(vals))} views ({len(vals)} snapshots)")
            if parts:
                lines.append(f"  {ht}: {' | '.join(parts)}")

        # Find clips still growing strongly at 7d+ (shares/saves signal longevity)
        late_rows = [r for r in rows if (r.get("hours_since_posted") or 0) >= 144]
        if late_rows:
            top_late = sorted(late_rows, key=lambda r: (r.get("shares") or 0) + (r.get("saves") or 0), reverse=True)[:3]
            if top_late:
                lines.append("\nCLIPS STILL GROWING AT 7 DAYS (high shares+saves = long-tail potential):")
                for r in top_late:
                    lines.append(f"  [{r.get('platform','?')}] \"{(r.get('hook') or '')[:70]}\" — {r.get('views',0)} views, {r.get('shares',0)} shares, {r.get('saves',0)} saves at {r.get('hours_since_posted','?')}h")

        return "\n".join(lines)
    except Exception as e:
        print(f"  Warning: growth analysis failed: {e}")
        return ""


def analyse_with_claude(
    rows: list[dict],
    decision_history: str,
    algorithm_research: str,
    timing_text: str,
    caption_text: str,
    follower_data: dict[str, int],
    content_analysis: str,
    growth_analysis: str,
    youtube_analytics: str = "",
    goal_progress: str = "",
    variant_analysis: str = "",
) -> str:
    """Send all performance data to Claude for strategic analysis."""
    if not rows:
        return "No synced performance data available yet."

    # Group by platform, sorted by views
    by_platform: dict[str, list] = defaultdict(list)
    for r in rows:
        by_platform[r["platform"]].append(r)
    for p in by_platform:
        by_platform[p].sort(key=lambda x: -x["views"])

    top    = rows[:20]
    bottom = rows[-10:] if len(rows) >= 15 else []

    dataset_text = "TOP PERFORMING CLIPS (ALL PLATFORMS, sorted by views):\n"
    for i, r in enumerate(top, 1):
        dataset_text += (
            f"{i}. [{r['platform'].upper()}] {r['views']} views | "
            f"{r['engagement_rate']}% eng | {r['comments']} comments | "
            f"{r['shares']} shares | {r['duration_seconds']}s | "
            f"Hook: \"{r['hook']}\"\n"
        )

    if bottom:
        dataset_text += "\nLOWEST PERFORMING CLIPS:\n"
        for i, r in enumerate(bottom, 1):
            dataset_text += (
                f"{i}. [{r['platform'].upper()}] {r['views']} views | "
                f"{r['engagement_rate']}% eng | {r['duration_seconds']}s | "
                f"Hook: \"{r['hook']}\"\n"
            )

    dataset_text += "\nTOP 5 PER PLATFORM:\n"
    for platform in ["tiktok", "instagram", "youtube", "facebook"]:
        if platform not in by_platform:
            continue
        dataset_text += f"\n{platform.upper()}:\n"
        for i, r in enumerate(by_platform[platform][:5], 1):
            dataset_text += (
                f"  {i}. {r['views']} views | {r['engagement_rate']}% eng | "
                f"{r['comments']} comments | {r['shares']} shares | "
                f"{r['duration_seconds']}s | Hook: \"{r['hook']}\"\n"
            )

    platform_summary: dict[str, dict] = {}
    for r in rows:
        p = r["platform"]
        if p not in platform_summary:
            platform_summary[p] = {"views": [], "engagement": [], "comments": [], "shares": []}
        platform_summary[p]["views"].append(r["views"])
        platform_summary[p]["engagement"].append(r["engagement_rate"])
        platform_summary[p]["comments"].append(r["comments"])
        platform_summary[p]["shares"].append(r["shares"])

    platform_text = "\nPLATFORM AVERAGES:\n"
    for p, d in platform_summary.items():
        avg_v = round(sum(d["views"]) / len(d["views"]))
        avg_e = round(sum(d["engagement"]) / len(d["engagement"]), 2)
        avg_c = round(sum(d["comments"]) / len(d["comments"]), 1)
        avg_s = round(sum(d["shares"]) / len(d["shares"]), 1)
        platform_text += (
            f"  {p.upper()}: avg {avg_v} views, {avg_e}% eng, "
            f"{avg_c} comments, {avg_s} shares ({len(d['views'])} clips)\n"
        )

    follower_text = ""
    if follower_data:
        follower_text = "\nCURRENT FOLLOWER COUNTS:\n"
        for platform, count in follower_data.items():
            follower_text += f"  {platform.upper()}: {count:,} followers\n"

    decision_block   = f"\n{decision_history}\n"                                              if decision_history   else ""
    timing_block     = f"\n{timing_text}\n"                                                   if timing_text        else ""
    caption_block    = f"\n{caption_text}\n"                                                  if caption_text       else ""
    algorithm_block  = f"\nPLATFORM ALGORITHM RESEARCH (current state):\n{algorithm_research}\n" if algorithm_research else ""
    content_block    = f"\n{content_analysis}\n"                                              if content_analysis   else ""
    growth_block     = f"\n{growth_analysis}\n"                                               if growth_analysis    else ""
    youtube_block    = f"\n{youtube_analytics}\n"                                             if youtube_analytics  else ""
    goal_block       = f"\n{goal_progress}\n"                                                 if goal_progress      else ""
    variant_block    = f"\n{variant_analysis}\n"                                              if variant_analysis   else ""

    prompt = f"""You are an expert social media strategist and content agent for the Konnected Minds Podcast (Ghana-based business/entrepreneurship channel posting short-form clips to TikTok, Instagram Reels, YouTube Shorts, and Facebook Reels).

GOAL (this is what every judgement below must be checked against): {GOAL}

Your role is not just to analyse past performance — you must act as a smart, proactive social media manager who understands what it takes to grow an audience, increase followers, and build real reach, not just post-level engagement.

Below is everything you need to know:

{dataset_text}
{platform_text}{follower_text}{goal_block}{variant_block}{decision_block}{timing_block}{caption_block}{content_block}{growth_block}{youtube_block}{algorithm_block}

Write a WEEKLY STRATEGY REVIEW that the clip selection AI will read before picking and captioning clips from a new episode. This brief must make the AI smarter — not just reactive to past data, but genuinely strategic about growth toward the GOAL above.

Structure your brief around these sections:

1. HOOK PATTERNS — What opening lines drive the most views on this channel? Quote specific hook examples from the top performers. Rank hook types.

2. TOPIC PATTERNS — Which subject areas consistently outperform? Which consistently underperform? Be specific about Ghana/Africa-relevant topics vs generic ones.

3. DURATION SWEET SPOT — What length actually works best for this channel based on the data? Does it vary by platform?

4. PLATFORM STRATEGY — For each platform (TikTok, Instagram, YouTube, Facebook), write a specific strategy combining: what this channel's data shows works, what the current algorithm rewards, and what caption tone/style/length to use. Make it specific to THIS channel and THIS audience.

5. CLIP CUTTING QUALITY — Based on the content analysis, what structural patterns separate the clips that perform best? What does a strong clip opening, middle, and ending look like for this channel? What should the AI prioritise when choosing exactly where to START and STOP a cut? What types of content create saves, shares, and return viewers? How should the channel approach each platform differently for audience building?

6. GROWTH TACTICS — Beyond individual clip views, what should the AI prioritise to grow followers and build a real audience? What content creates saves, shares, and return viewers?

7. POSTING TIMING — Based on the timing data, when should content be posted for maximum reach?

8. WHAT TO AVOID — Specific traits of the lowest performers. What hooks, topics, caption styles, and clip structures are actively hurting performance?

9. ACTIONABLE RULES — 10 to 12 specific, concrete rules for clip selection, cutting decisions, AND caption writing. Each rule must be direct and immediately applicable (e.g. "On TikTok, always prioritise confession hooks — they average 2x the channel's TikTok mean. Never use advice hooks as openers on TikTok."). These rules are fed automatically into the clip-selection AI's prompt every run.

10. IS THE CURRENT STRATEGY WORKING? — Using the GOAL PROGRESS data above (this week vs last week, per platform), give an explicit verdict per platform: growing, flat, or declining, with the actual numbers as evidence. Do not soften this with vibes — if a platform is flat or declining, say so plainly. If there isn't enough week-over-week data yet, say that plainly too instead of guessing.

11. FOR TED — RECOMMENDED ACTIONS (human review, not auto-applied) — This section is fundamentally different from section 9: rules in section 9 are read automatically by the clip-selection AI every run. This section is NOT auto-applied to anything — it is written for a human to read and decide on. Write 3-5 specific, concrete proposed changes to the pipeline or strategy itself (not clip-picking rules), each grounded in real evidence from above. Examples of the KIND of thing that belongs here (use real numbers from the data, not these exact examples): "Consider shifting the caption A/B split toward the dynamic style — it's averaging Nx more views after M posts, though the sample is still small." / "TikTok has been flat for 2 weeks running despite steady posting — consider investigating before increasing volume there." / "Engagement per post is declining as posting volume increases — consider reducing cadence." If nothing rises to the level of a real recommendation yet (not enough data, or nothing conclusive), say so plainly rather than inventing filler.
    FORMATTING FOR THIS SECTION ONLY: do NOT start each recommendation with a number and period (e.g. "1. ", "2. ") -- write each as its own short bolded label followed by a colon and the explanation, as a plain paragraph (e.g. "**Facebook caption format:** ..."). This is a strict technical requirement of the rendering system this feeds into, not a style preference.

Be specific, data-driven, and opinionated. Reference real numbers and real hook examples from the data. Write as if you are briefing a junior social media manager — clear, direct, no fluff.

LENGTH BUDGET — this is an 11-section brief and you have a finite output budget. Keep sections 1-8 tight (roughly 100-150 words of prose each; data tables/examples don't count against this). Sections 9, 10, and 11 are the most operationally important parts of this weekly review — never let earlier sections run long enough that you run out of room before reaching them. If you're running low on space, compress sections 1-8 further rather than cutting 9-11 short."""

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Run the full weekly review now, bypassing the Monday gate")
    args = parser.parse_args()

    print("Fetching analytics...")
    zernio_map = fetch_zernio_all_posts()
    print(f"  {len(zernio_map)} posts fetched.")

    print("Fetching follower counts...")
    follower_data = fetch_follower_counts()

    print("Fetching clip metadata from Supabase...")
    clips = fetch_clip_metadata()
    print(f"  {len(clips)} posted clips with hooks found.")

    print("Updating clip selection outcomes...")
    update_selection_outcomes(zernio_map)

    print("Building performance dataset...")
    rows = build_performance_dataset(clips, zernio_map)
    print(f"  {len(rows)} clips with synced analytics.")

    if len(rows) < 5:
        print("  Not enough synced data yet — skipping analysis.")
        return

    # Full weekly strategy review runs Monday mornings (per the intended cadence
    # of this being a *weekly* review, not a running stat dump) -- unless this
    # is the very first review ever (bootstrap) or --force is passed.
    is_monday = datetime.now(timezone.utc).weekday() == 0
    try:
        existing = sb.table("channel_intelligence").select("stats,updated_at").eq("id", "singleton").maybe_single().execute()
        has_existing_review = bool(existing.data)
    except Exception:
        has_existing_review = False

    if not args.force and has_existing_review and not is_monday:
        print(f"  Not Monday (today is {datetime.now(timezone.utc).strftime('%A')}) and a review already "
              f"exists -- skipping full weekly review. Use --force to run anyway.")
        return
    print(f"  Running full weekly review "
          f"({'forced' if args.force else 'first-ever review' if not has_existing_review else 'Monday'}).")

    # Gather all intelligence inputs
    print("Fetching YouTube Analytics...")
    youtube_analytics = fetch_youtube_analytics()

    print("Researching platform algorithms...")
    algorithm_research = research_platform_algorithms()

    print("Building decision history...")
    decision_history = build_decision_history(zernio_map)

    print("Building caption style A/B analysis...")
    variant_analysis = build_variant_analysis(zernio_map)

    print("Building goal progress (this week vs last week)...")
    goal_progress = build_goal_progress(rows)

    print("Building timing analysis...")
    timing_text = build_timing_analysis(rows)

    print("Building caption analysis...")
    caption_text = build_caption_analysis(rows)

    print("Building clip content analysis...")
    content_analysis = build_content_analysis(zernio_map)

    print("Building view growth analysis...")
    growth_analysis = build_growth_analysis()

    stats = {
        "clips_analysed":  len(rows),
        "total_views":     sum(r["views"] for r in rows),
        "avg_engagement":  round(sum(r["engagement_rate"] for r in rows) / len(rows), 2),
        "best_hook":       rows[0]["hook"] if rows else "",
        "best_views":      rows[0]["views"] if rows else 0,
        "follower_counts": follower_data,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

    print("Sending to Claude for strategic analysis...")
    summary = analyse_with_claude(
        rows, decision_history, algorithm_research, timing_text, caption_text,
        follower_data, content_analysis, growth_analysis, youtube_analytics,
        goal_progress=goal_progress, variant_analysis=variant_analysis,
    )
    print("  Analysis complete.")
    print("\n--- INTELLIGENCE BRIEF PREVIEW ---")
    print(summary[:600] + ("..." if len(summary) > 600 else ""))
    print("---\n")

    save_intelligence(summary, stats)
    print(f"Done. {len(rows)} clips analysed, intelligence saved.")


if __name__ == "__main__":
    main()
