#!/usr/bin/env python3
"""
publish_youtube.py - upload rendered clips to YouTube as Shorts.

YouTube is the ONE leg of this pipeline that can legitimately be automated.
Uploads bill to their own quota bucket at 100 calls/day, so volume is not
the constraint.

TikTok cannot be automated this way. Unaudited API clients can only post
SELF_ONLY (private), capped at 5 users per 24h, with accounts required to
be private at the time of posting. Lifting that needs a full app audit you
would not pass as a solo operator. Post TikTok by hand or schedule in
TikTok Studio (10 days ahead).

Setup:
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3"
  2. OAuth consent screen -> External -> add yourself as a test user
  3. Credentials -> OAuth client ID -> Desktop app -> download as client_secret.json
  4. python3 publish_youtube.py out/ep42/ep42_03.mp4 --title "..." --schedule "2026-09-16T14:00:00Z"

First run opens a browser once, then caches token.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
ROOT = Path(__file__).resolve().parent


def service():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit("Missing deps. Run: pip install google-api-python-client google-auth-oauthlib")

    tok, sec = ROOT / "token.json", ROOT / "client_secret.json"
    creds = None
    if tok.exists():
        creds = Credentials.from_authorized_user_file(str(tok), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not sec.exists():
                sys.exit(f"Put your OAuth client JSON at {sec}")
            creds = InstalledAppFlow.from_client_secrets_file(str(sec), SCOPES).run_local_server(port=0)
        tok.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload(path, title, description, tags, schedule=None, private=False):
    from googleapiclient.http import MediaFileUpload

    status = {"selfDeclaredMadeForKids": False}
    if schedule:
        status["privacyStatus"] = "private"
        status["publishAt"] = schedule          # RFC3339, e.g. 2026-09-16T14:00:00Z
    else:
        status["privacyStatus"] = "private" if private else "public"

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": tags[:15],
            "categoryId": "24",
        },
        "status": status,
    }

    media = MediaFileUpload(str(path), chunksize=-1, resumable=True, mimetype="video/mp4")
    req = service().videos().insert(part="snippet,status", body=body, media_body=media)

    resp = None
    while resp is None:
        prog, resp = req.next_chunk()
        if prog:
            print(f"  {int(prog.progress() * 100)}%", end="\r", flush=True)
    vid = resp["id"]
    print(f"  uploaded: https://youtube.com/shorts/{vid}")
    return vid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--schedule", default=None, help="RFC3339 UTC, e.g. 2026-09-16T14:00:00Z")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    if not Path(a.video).exists():
        sys.exit(f"not found: {a.video}")

    desc = a.description
    sidecar = Path(a.video).with_suffix(".txt")
    if not desc and sidecar.exists():
        desc = sidecar.read_text()

    # #Shorts is no longer required for Shorts classification (aspect ratio and
    # duration decide it) but it costs nothing.
    if "#Shorts" not in desc:
        desc = desc.rstrip() + "\n\n#Shorts"

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    upload(a.video, a.title, desc, tags, a.schedule, a.private)


if __name__ == "__main__":
    main()
