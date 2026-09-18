"""
Google Photos retrieval research -- data collection (free, runs locally).

Pulls public reviews/discussions from Play Store, App Store, and Reddit
about people struggling to find old/vaguely-remembered photos, and saves
them as CSVs in ./data/. No API keys needed for this step -- everything
here hits public pages.

Usage:
    python -m venv venv
    source venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    python collect_reviews.py

Notes on sources (as of 2026):
  - Play Store: google-play-scraper works unauthenticated, unchanged.
  - App Store: the app-store-scraper PyPI package is unmaintained and pulls
    in an ancient urllib3 that breaks on modern Python, so this script talks
    directly to Apple's public iTunes RSS review feed instead (that package
    was just a thin wrapper around the same feed). Apple caps this feed at
    10 pages (~500 reviews) per country/sort combo, so we pull several
    English-speaking storefronts and both sort orders and de-dupe.
  - Reddit: old.reddit.com's json endpoints and www.reddit.com's json
    endpoints are both gated behind login/bot-detection now. The Atom
    (.rss) endpoints on www.reddit.com still work without auth, so this
    script uses those instead, with a rate limiter that honors Reddit's
    x-ratelimit-* response headers (anonymous access is throttled to
    roughly one request per 20-30 seconds) plus exponential backoff on
    429/403/5xx. This means Reddit collection is slow (expect 45-90
    minutes) -- that's Reddit's anonymous rate limit, not a bug.

Next step (free, no separate API key): upload data/raw_combined.csv to
your Claude chat and ask for it to be tagged/clustered -- that AI step
runs on your existing Claude access instead of a paid API.
"""
import html
import os
import re
import time
import xml.etree.ElementTree as ET

import requests
import pandas as pd
from tqdm import tqdm

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Play Store reviews
# ---------------------------------------------------------------------------
def collect_play_store(app_id="com.google.android.apps.photos", target=3000):
    from google_play_scraper import reviews, Sort

    out = []
    for sort in (Sort.NEWEST, Sort.MOST_RELEVANT):
        token = None
        fetched = 0
        while fetched < target // 2:
            batch, token = reviews(
                app_id, lang="en", country="us", sort=sort,
                count=200, continuation_token=token
            )
            if not batch:
                break
            for r in batch:
                out.append({
                    "source": "play_store", "id": r["reviewId"], "date": str(r["at"]),
                    "rating": r["score"], "text": r["content"],
                    "url": f"https://play.google.com/store/apps/details?id={app_id}",
                })
            fetched += len(batch)
            if token is None:
                break
            time.sleep(0.5)
    df = pd.DataFrame(out).drop_duplicates(subset="id")
    df.to_csv(os.path.join(OUT_DIR, "raw_play_store.csv"), index=False)
    print(f"[play store] {len(df)} reviews -> data/raw_play_store.csv")
    return df


# ---------------------------------------------------------------------------
# 2. App Store reviews (Apple's public iTunes RSS review feed, no auth)
# ---------------------------------------------------------------------------
APP_STORE_ID = 962194608  # "Google Photos: Backup & Edit" (bundle id com.google.photos)
APP_STORE_COUNTRIES = ["us", "gb", "ca", "au", "in", "ie"]
ITUNES_HEADERS = {"User-Agent": "Mozilla/5.0 (gphotos-retrieval-research/0.2)"}


def _itunes_get(url, tries=5):
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=ITUNES_HEADERS, timeout=20)
        except requests.RequestException:
            time.sleep(min(30, 3 * (attempt + 1)))
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(30, 3 * (2 ** attempt)))
            continue
        return None  # e.g. 400 once we're past the last available page
    return None


def collect_app_store(app_id=APP_STORE_ID, target=1500):
    out = {}
    combos = [(c, s) for c in APP_STORE_COUNTRIES for s in ("mostrecent", None)]
    for country, sortby in tqdm(combos, desc="app store country/sort"):
        for page in range(1, 11):
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/"
            if sortby:
                url += f"sortby={sortby}/"
            url += f"page={page}/json"
            resp = _itunes_get(url)
            if resp is None:
                break
            try:
                entries = resp.json().get("feed", {}).get("entry", [])
            except ValueError:
                break
            if not entries:
                break
            for e in entries:
                if "im:rating" not in e:
                    continue
                rid = e.get("id", {}).get("label")
                if not rid or rid in out:
                    continue
                title = e.get("title", {}).get("label", "")
                content = e.get("content", {}).get("label", "")
                out[rid] = {
                    "source": "app_store", "id": rid,
                    "date": e.get("updated", {}).get("label", ""),
                    "rating": e.get("im:rating", {}).get("label"),
                    "text": (title + "\n" + content).strip(),
                    "url": f"https://apps.apple.com/{country}/app/google-photos/id{app_id}",
                }
            time.sleep(1)
            if len(entries) < 50:
                break  # last page for this country/sort
        if len(out) >= target:
            break
    df = pd.DataFrame(list(out.values()))
    df.to_csv(os.path.join(OUT_DIR, "raw_app_store.csv"), index=False)
    print(f"[app store] {len(df)} reviews -> data/raw_app_store.csv")
    return df


# ---------------------------------------------------------------------------
# 3. Reddit (public Atom/.rss feeds, no login/API key needed)
# ---------------------------------------------------------------------------
REDDIT_HEADERS = {"User-Agent": "gphotos-retrieval-research/0.2 (student capstone project)"}
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
SEARCH_TERMS = [
    "google photos can't find", "google photos search not working", "lost photo google photos",
    "google photos remember but can't find", "google photos search sucks", "find old photo google photos",
    "google photos ai search", "google photos ask photos",
]
SUBREDDITS = ["GooglePhotos", "google", "androidapps", "photography"]
MAX_COMMENT_FETCHES = 45  # global bound on total runtime under Reddit's strict anon rate limit
COMMENTS_PER_COMBO = 3    # bound per (term, subreddit) so one popular term can't eat the whole budget

_rate_next_ok = [0.0]


def _reddit_get(url, params=None, tries=4):
    for attempt in range(tries):
        wait = _rate_next_ok[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue

        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if remaining is not None and reset is not None:
            try:
                if float(remaining) < 1:
                    _rate_next_ok[0] = time.time() + float(reset) + 1.0
                else:
                    _rate_next_ok[0] = time.time() + 1.5
            except ValueError:
                _rate_next_ok[0] = time.time() + 5
        else:
            _rate_next_ok[0] = time.time() + 5

        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 403, 500, 502, 503, 504):
            retry_after = resp.headers.get("retry-after")
            backoff = float(retry_after) if retry_after else min(45, 6 * (2 ** attempt))
            time.sleep(backoff)
            continue
        return None
    return None


def _clean_reddit_html(raw):
    if not raw:
        return ""
    text = raw.replace("<!-- SC_OFF -->", "").replace("<!-- SC_ON -->", "")
    text = re.sub(r"submitted by.*", "", text, flags=re.S)  # reddit's added footer on post entries
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def reddit_search(query, subreddit=None, limit=15):
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.rss"
        params = {"q": query, "restrict_sr": "1", "limit": limit, "sort": "relevance", "t": "all"}
    else:
        url = "https://www.reddit.com/search.rss"
        params = {"q": query, "limit": limit, "sort": "relevance", "t": "all"}
    resp = _reddit_get(url, params)
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    posts = []
    for entry in root.findall("a:entry", ATOM_NS):
        entry_id = entry.findtext("a:id", default="", namespaces=ATOM_NS)
        pid = entry_id.split("_")[-1] if entry_id else None
        link_el = entry.find("a:link", ATOM_NS)
        permalink = link_el.get("href") if link_el is not None else ""
        if not pid or not permalink:
            continue
        title = entry.findtext("a:title", default="", namespaces=ATOM_NS)
        content = entry.findtext("a:content", default="", namespaces=ATOM_NS)
        published = entry.findtext("a:published", default="", namespaces=ATOM_NS)
        posts.append({
            "id": pid, "title": title, "text": _clean_reddit_html(content),
            "permalink": permalink, "date": published,
        })
    return posts


def get_comments(permalink, limit=15):
    url = permalink if permalink.endswith("/") else permalink + "/"
    url += ".rss"
    resp = _reddit_get(url, {"limit": limit})
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []
    entries = root.findall("a:entry", ATOM_NS)[1:]  # entry 0 is the post itself
    comments = []
    for e in entries[:limit]:
        text = _clean_reddit_html(e.findtext("a:content", default="", namespaces=ATOM_NS))
        if text:
            comments.append(text)
    return comments


def collect_reddit():
    rows, seen_ids = [], set()
    comment_fetches = 0
    checkpoint_path = os.path.join(OUT_DIR, "raw_reddit.csv")
    for term in tqdm(SEARCH_TERMS, desc="reddit search terms"):
        for sub in [None] + SUBREDDITS:
            combo_comment_fetches = 0
            for post in reddit_search(term, sub):
                pid = post["id"]
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                body = (post["title"] + "\n" + post["text"]).strip()
                rows.append({
                    "source": "reddit_post", "id": pid, "date": post["date"],
                    "rating": None, "text": body, "url": post["permalink"],
                })
                if (len(post["text"]) > 40 and combo_comment_fetches < COMMENTS_PER_COMBO
                        and comment_fetches < MAX_COMMENT_FETCHES):
                    comment_fetches += 1
                    combo_comment_fetches += 1
                    for i, c in enumerate(get_comments(post["permalink"])):
                        rows.append({
                            "source": "reddit_comment", "id": f"{pid}_c{i}", "date": post["date"],
                            "rating": None, "text": c, "url": post["permalink"],
                        })
        # checkpoint after every search term so an interruption doesn't lose all progress
        pd.DataFrame(rows).drop_duplicates(subset="id").to_csv(checkpoint_path, index=False)
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    df.to_csv(checkpoint_path, index=False)
    print(f"[reddit] {len(df)} posts/comments -> data/raw_reddit.csv")
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_play = collect_play_store()
    df_app = collect_app_store()
    df_reddit = collect_reddit()

    df_all = pd.concat([df_play, df_app, df_reddit], ignore_index=True)
    df_all = df_all[df_all["text"].str.len() > 15].reset_index(drop=True)
    df_all.to_csv(os.path.join(OUT_DIR, "raw_combined.csv"), index=False)

    print(f"\nTotal usable items: {len(df_all)}")
    print(df_all["source"].value_counts())
    print(f"\nAll set. Upload data/raw_combined.csv to your Claude chat next for the tagging/analysis step.")
