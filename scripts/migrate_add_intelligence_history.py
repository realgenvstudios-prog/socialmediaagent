#!/usr/bin/env python3
"""
One-time migration: creates the intelligence_history table, an insert-only
audit trail of every weekly strategy review (channel_intelligence only ever
keeps the latest one). Safe to run multiple times (uses IF NOT EXISTS).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]

SQL = """
CREATE TABLE IF NOT EXISTS intelligence_history (
    id         UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    summary    TEXT,
    stats      JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

resp = requests.post(
    f"{SUPABASE_URL}/rest/v1/rpc/exec",
    headers={
        "Authorization": f"Bearer {SERVICE_KEY}",
        "apikey": SERVICE_KEY,
        "Content-Type": "application/json",
    },
    json={"sql": SQL},
    timeout=15,
)

if resp.ok:
    print("Migration successful.")
else:
    print(f"Could not run via RPC ({resp.status_code}): {resp.text[:200]}")
    print()
    print("Run this SQL manually in Supabase SQL Editor:")
    print(SQL)
