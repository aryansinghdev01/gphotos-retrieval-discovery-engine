"""Groq-based LLM classifier for the Google Photos retrieval-failure taxonomy.

Batches rows into single prompts (structured JSON array response) to conserve
free-tier requests, with rate-limit-aware backoff.
"""
import json
import os
import time

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
MAX_TEXT_WORDS = 130  # truncate very long rows to keep batch token cost predictable

CLUSTER_TAXONOMY = """\
1. "Browsing/visual structure removed or broken" — existing tags: removed browsing structure \
hurts findability, broken chronological ordering hurts browsing, no nested folders/albums for \
organization, no traditional file/folder navigation, favorites/quick-access broken forces manual \
browsing, folder-to-album mismatch, lost organizational tools reduce findability, wants easier \
retrieval of most-recent items
2. "Search itself is broken" — existing tags: keyword search returns no results for existing \
photo, OCR/text-in-image search broke, date search stopped working, new AI ask-search \
underperforms old keyword search, search completely broken after app update, search degraded \
needs many more clarifying steps
3. "Content exists but isn't indexed where expected" — existing tags: photos backed up but \
missing from gallery, synced library incomplete/inconsistent across devices, synced library \
incomplete/inconsistent in gallery integrations, edited photos don't sync to sharing pickers, \
face grouping broken for recent photos, face grouping/labeling broken, share picker doesn't \
surface recently added photos, photos surface in auto memories but not in gallery/search, \
AI-edited results hard to relocate after creation
4. "Fragmentation / apparent loss" — existing tags: photos vanish/hidden with no clear \
explanation, fragmented across multiple accounts no unified search, photo appears lost wants it \
restored
"""

SYSTEM_PROMPT = f"""You are classifying user reviews/posts about Google Photos for a UX \
research study on photo RETRIEVAL failures — anything that stops someone from finding, \
browsing to, or relocating a photo/video they know exists.

is_retrieval_case is true if the text describes ANY of these:
(a) actively trying and failing/struggling to find or relocate a specific remembered photo/video \
via search, scrolling, or browsing;
(b) a browsing/organizational feature that was removed, changed, or is broken in a way that \
impairs finding or organizing photos in general — e.g. no nested folders/albums, broken or \
removed chronological ordering, lost folder/file navigation, broken favorites/quick-access, \
folder-to-album mismatches — even without naming one specific remembered photo;
(c) content that exists but doesn't show up where the person expects it (sync issues, edited \
photos missing from sharing pickers, broken face grouping, items backed up but absent from the \
gallery);
(d) photos/videos that seem to have vanished, gotten lost, or fragmented across accounts with no \
explanation.

It is false for general complaints unrelated to finding/browsing/organizing photos: pricing, \
storage cost, privacy, ads, unrelated feature requests, plain praise, or bugs unconnected to \
locating or organizing content (e.g. crashes, slow uploads, login issues, editing tools).

If is_retrieval_case is true, classify into exactly one of these 4 clusters and pick the closest \
matching opportunity_tag from that cluster's existing tags below, OR propose a new short \
opportunity_tag (3-8 words, same style) if none of the existing ones fit well:

{CLUSTER_TAXONOMY}

Input is a JSON array of rows: [{{"i": <int>, "text": "..."}}, ...].
Output STRICT JSON: {{"results": [{{"i": <int>, "is_retrieval_case": bool, "cluster": <1-4 or \
null>, "opportunity_tag": <string or null>, "photo_type": <string or "">, \
"what_they_remember": [<string>], "what_they_forgot": [<string>], "search_attempt": <string or \
"">, "outcome": <"success"|"failure"|"unclear" or "">, "workaround": <string or "">}}, ...]}}
One result object per input row, "i" must match. If is_retrieval_case is false, set cluster and \
opportunity_tag to null and other fields to empty string/list. Output only the JSON object, \
nothing else."""


class RateLimitTracker:
    def __init__(self):
        self.remaining_tokens = None
        self.reset_tokens_seconds = None

    def update(self, headers):
        rt = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
        if rt is not None:
            self.remaining_tokens = int(float(rt))
        if reset is not None:
            self.reset_tokens_seconds = _parse_reset(reset)

    def wait_if_needed(self, estimated_tokens):
        if self.remaining_tokens is not None and self.remaining_tokens < estimated_tokens:
            wait = (self.reset_tokens_seconds or 5) + 1
            time.sleep(wait)


def _parse_reset(s):
    # formats like "3.022s", "1m26.4s"
    s = s.strip()
    total = 0.0
    if "m" in s:
        m, s = s.split("m", 1)
        total += float(m) * 60
    s = s.rstrip("s")
    if s:
        total += float(s)
    return total


_tracker = RateLimitTracker()


def classify_batch(rows, tries=5):
    """rows: list of {"i": int, "text": str}. Returns dict i -> result dict, or {} on failure."""
    payload_rows = []
    for r in rows:
        text = " ".join(r["text"].split()[:MAX_TEXT_WORDS])
        payload_rows.append({"i": r["i"], "text": text})

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload_rows)},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
        "temperature": 0,
    }

    est_tokens = len(json.dumps(payload_rows)) // 3 + len(SYSTEM_PROMPT) // 3 + 800
    _tracker.wait_if_needed(est_tokens)

    for attempt in range(tries):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
        except requests.RequestException:
            time.sleep(min(30, 5 * (attempt + 1)))
            continue

        _tracker.update(resp.headers)

        if resp.status_code == 200:
            try:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict):
                    items = parsed.get("results")
                    if items is None:
                        # model sometimes wraps the array under a different key
                        list_values = [v for v in parsed.values() if isinstance(v, list)]
                        items = list_values[0] if list_values else []
                else:
                    items = []
                out = {}
                for item in items:
                    if isinstance(item, dict) and "i" in item:
                        out[item["i"]] = item
                return out
            except (KeyError, ValueError, json.JSONDecodeError, TypeError) as e:
                print(f"  [classify_batch] parse failure on attempt {attempt+1}: {e}: "
                      f"{resp.text[:200]!r}")
                continue
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            wait = float(retry_after) if retry_after else _parse_reset(
                resp.headers.get("x-ratelimit-reset-tokens", "10s")
            )
            print(f"  [classify_batch] 429 on attempt {attempt+1}, waiting {wait+1:.0f}s")
            time.sleep(wait + 1)
            continue
        if resp.status_code in (500, 502, 503, 504):
            wait = min(30, 5 * (2 ** attempt))
            print(f"  [classify_batch] {resp.status_code} on attempt {attempt+1}, waiting {wait}s")
            time.sleep(wait)
            continue
        # 4xx other than 429: not retryable
        print(f"  [classify_batch] non-retryable HTTP {resp.status_code}: {resp.text[:200]!r}")
        return {}
    print(f"  [classify_batch] gave up after {tries} attempts")
    return {}
