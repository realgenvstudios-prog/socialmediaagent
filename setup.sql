-- Run this in your Supabase SQL Editor
-- supabase.com → your project → SQL Editor → New query → paste → Run

-- Tracks which YouTube videos have been processed
CREATE TABLE IF NOT EXISTS processed_videos (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    video_id     TEXT        UNIQUE NOT NULL,
    video_title  TEXT        NOT NULL,
    channel_id   TEXT        NOT NULL,
    clip_count   INTEGER     DEFAULT 0,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Queue of clips waiting to be posted, one row per clip per platform
CREATE TABLE IF NOT EXISTS clip_queue (
    id             UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    video_id       TEXT        NOT NULL,
    clip_index     INTEGER     NOT NULL,
    storage_path   TEXT        NOT NULL,
    public_url     TEXT        NOT NULL,
    caption        TEXT        NOT NULL,
    hook           TEXT,
    platform       TEXT        NOT NULL CHECK (platform IN ('instagram', 'tiktok', 'youtube', 'facebook')),
    status         TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'posted', 'failed')),
    zernio_post_id TEXT,
    posted_at      TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Fast queue lookups (platform + status + order)
CREATE INDEX IF NOT EXISTS idx_clip_queue_pending
    ON clip_queue(platform, status, created_at);

-- Pipeline error log (written by GitHub Actions on workflow failure)
CREATE TABLE IF NOT EXISTS pipeline_errors (
    id         UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    workflow   TEXT        NOT NULL,
    run_url    TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Posting schedule config (optional — post_clips.py reads this)
-- Insert one row: { "times": ["09:00", "13:00", "18:00"] }
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO settings (key, value)
VALUES ('schedule', '{"times": ["09:00", "13:00", "18:00"]}')
ON CONFLICT (key) DO NOTHING;

-- Saved transcripts so we never re-transcribe the same video
CREATE TABLE IF NOT EXISTS video_transcripts (
    video_id     TEXT PRIMARY KEY,
    transcript   TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Claude's clip plan — saved immediately after Claude responds
-- so the pipeline can resume from here if it crashes mid-download
CREATE TABLE IF NOT EXISTS video_clip_plans (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    video_id     TEXT        NOT NULL,
    clip_index   INTEGER     NOT NULL,
    start_seconds FLOAT      NOT NULL,
    end_seconds  FLOAT       NOT NULL,
    caption      TEXT        NOT NULL,
    hook         TEXT,
    status       TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'failed')),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (video_id, clip_index)
);

-- Per-clip selection log — records every clip Claude selected and its eventual performance
-- Used by update_intelligence.py to learn which hook types, topics, and durations perform best
CREATE TABLE IF NOT EXISTS clip_selection_log (
    id                  UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    video_id            TEXT        NOT NULL,
    clip_index          INTEGER     NOT NULL,
    hook                TEXT,
    hook_type           TEXT,
    topic_category      TEXT,
    duration_seconds    FLOAT,
    views               INTEGER     DEFAULT 0,
    likes               INTEGER     DEFAULT 0,
    engagement_rate     FLOAT       DEFAULT 0,
    performance_tier    TEXT,
    clip_transcript     TEXT,
    caption_variant     TEXT,
    analytics_updated_at TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (video_id, clip_index)
);

-- Daily time-series snapshots of clip engagement (one row per clip per day)
-- Tracks how views, shares, saves grow over time — reveals long-tail vs flash-in-pan clips
CREATE TABLE IF NOT EXISTS clip_performance (
    id                  UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    clip_queue_id       UUID        NOT NULL,
    video_id            TEXT        NOT NULL,
    clip_index          INTEGER     NOT NULL,
    zernio_post_id      TEXT,
    platform            TEXT        NOT NULL,
    hours_since_posted  INTEGER,
    views               INTEGER     DEFAULT 0,
    impressions         INTEGER     DEFAULT 0,
    reach               INTEGER     DEFAULT 0,
    likes               INTEGER     DEFAULT 0,
    comments            INTEGER     DEFAULT 0,
    shares              INTEGER     DEFAULT 0,
    saves               INTEGER     DEFAULT 0,
    clicks              INTEGER     DEFAULT 0,
    engagement_rate     FLOAT       DEFAULT 0,
    measured_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clip_performance_clip
    ON clip_performance(clip_queue_id, measured_at);

-- Channel intelligence brief — single row updated daily by update_intelligence.py
-- Injected into every Claude clip selection prompt via process_video.py
CREATE TABLE IF NOT EXISTS channel_intelligence (
    id         TEXT        PRIMARY KEY DEFAULT 'singleton',
    summary    TEXT,
    stats      JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
