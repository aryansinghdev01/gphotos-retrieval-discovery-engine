"""Groq-based LLM classifier for the Google Photos retrieval-failure taxonomy.

Batches rows into single prompts (structured JSON response) to conserve free-tier
requests. opportunity_tag is constrained to the closed set of tags used in the
human-tagged findings (or "other — <short tag>"); the cluster is derived from the
tag, so counts roll up into the 4 existing clusters.
"""
import difflib
import json
import os
import time

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TEXT_WORDS = 130  # truncate very long rows to keep batch token cost predictable

CLUSTER_NAMES = {
    1: "Browsing/visual structure removed or broken",
    2: "Search itself is broken",
    3: "Content exists but isn't indexed where expected",
    4: "Fragmentation / apparent loss",
}
_NAME_TO_NUM = {v: k for k, v in CLUSTER_NAMES.items()}

_FINDINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "discovery_findings.json")


def _load_closed_tag_set():
    """tag -> cluster number, taken from the human-tagged findings."""
    with open(_FINDINGS_PATH) as f:
        findings = json.load(f)
    tag_to_cluster = {}
    for cluster in findings["clusters"]:
        for case in cluster["cases"]:
            tag_to_cluster[case["opportunity_tag"]] = _NAME_TO_NUM[cluster["name"]]
    return tag_to_cluster


TAG_TO_CLUSTER = _load_closed_tag_set()
_TAG_LOOKUP = {t.casefold().strip(): t for t in TAG_TO_CLUSTER}


def _taxonomy_text():
    lines = []
    for num, name in CLUSTER_NAMES.items():
        tags = [t for t, c in TAG_TO_CLUSTER.items() if c == num]
        lines.append(f'Cluster {num}: "{name}"')
        lines.extend(f"  - {t}" for t in sorted(tags))
    return "\n".join(lines)


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

If is_retrieval_case is true, opportunity_tag MUST be copied EXACTLY (character for character) \
from this closed list — do not reword, shorten, or invent tags. Pick the closest one:

{_taxonomy_text()}

Only if genuinely NONE of the listed tags fits, set opportunity_tag to "other — <3-8 word tag>" \
and set "cluster" to the closest cluster number (1-4). When you pick a listed tag, set "cluster" \
to null (it is derived from the tag).

Input is a JSON array of rows: [{{"i": <int>, "text": "..."}}, ...].
Output STRICT JSON: {{"results": [{{"i": <int>, "is_retrieval_case": bool, "opportunity_tag": \
<exact listed tag | "other — ..." | null>, "cluster": <1-4 or null>, "photo_type": <string or \
"">, "what_they_remember": [<string>], "what_they_forgot": [<string>], "search_attempt": \
<string or "">, "outcome": <"success"|"failure"|"unclear" or "">, "workaround": <string or \
"">}}, ...]}}
One result object per input row, "i" must match. If is_retrieval_case is false, set cluster and \
opportunity_tag to null and other fields to empty string/list. Output only the JSON object, \
nothing else."""


def normalize_result(item):
    """Enforce the closed tag set and derive the cluster from the tag."""
    if not item.get("is_retrieval_case"):
        item["cluster"] = None
        item["opportunity_tag"] = None
        item["tag_in_closed_set"] = None
        return item

    tag = (item.get("opportunity_tag") or "").strip()
    canonical = _TAG_LOOKUP.get(tag.casefold())
    if canonical is None and tag:
        # tolerate trivial wording drift (punctuation/case), not real rewording
        close = difflib.get_close_matches(tag.casefold(), list(_TAG_LOOKUP), n=1, cutoff=0.92)
        canonical = _TAG_LOOKUP[close[0]] if close else None

    if canonical is not None:
        item["opportunity_tag"] = canonical
        item["cluster"] = TAG_TO_CLUSTER[canonical]
        item["tag_in_closed_set"] = True
    else:
        try:
            item["cluster"] = int(item.get("cluster"))
        except (TypeError, ValueError):
            item["cluster"] = None
        if not tag.casefold().startswith("other"):
            item["opportunity_tag"] = f"other — {tag}" if tag else "other"
        item["tag_in_closed_set"] = False
    return item


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
            time.sleep((self.reset_tokens_seconds or 5) + 1)


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

MIN_CALL_INTERVAL = float(os.environ.get("GROQ_MIN_INTERVAL", "20"))   # floor between call attempts; the free tier throttles bursts
MAX_BACKOFF = 300        # never sleep longer than this on a single retry
_last_call_at = [0.0]
last_error = [""]        # reason for the most recent failure, for failure logging


def _enforce_min_interval():
    wait = MIN_CALL_INTERVAL - (time.time() - _last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    _last_call_at[0] = time.time()


def _backoff(attempt, hint=None):
    """Exponential backoff (5s, 10s, 20s, ...) unless the server gave a wait hint."""
    wait = hint if hint is not None else 5 * (2 ** attempt)
    time.sleep(min(MAX_BACKOFF, wait) + 1)


def classify_batch(rows, tries=8):
    """rows: list of {"i": int, "text": str}. Returns dict i -> normalized result.

    Retries with exponential backoff on 429/5xx/timeouts/unparseable output; honors the
    server's retry-after (capped). May return a partial dict if the model omitted rows —
    callers must retry the missing ones. Never raises; on total failure returns {} and
    sets last_error[0].
    """
    payload_rows = [{"i": r["i"], "text": " ".join(r["text"].split()[:MAX_TEXT_WORDS])} for r in rows]
    wanted = {r["i"] for r in rows}

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload_rows)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = "low"

    est_tokens = len(json.dumps(payload_rows)) // 3 + len(SYSTEM_PROMPT) // 3 + 800
    _tracker.wait_if_needed(est_tokens)

    for attempt in range(tries):
        _enforce_min_interval()
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=90,
            )
        except requests.RequestException as e:
            last_error[0] = f"network error: {type(e).__name__}"
            print(f"  [classify_batch] {last_error[0]} (attempt {attempt+1}/{tries})")
            _backoff(attempt)
            continue

        _tracker.update(resp.headers)

        if resp.status_code == 200:
            try:
                parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict):
                    items = parsed.get("results")
                    if items is None:
                        list_values = [v for v in parsed.values() if isinstance(v, list)]
                        items = list_values[0] if list_values else []
                else:
                    items = []
                out = {}
                for item in items:
                    if isinstance(item, dict) and item.get("i") in wanted:
                        out[item["i"]] = normalize_result(item)
                if out:
                    return out
                last_error[0] = "response parsed but contained no usable results"
            except (KeyError, ValueError, TypeError) as e:
                last_error[0] = f"unparseable response: {type(e).__name__}"
            print(f"  [classify_batch] {last_error[0]} (attempt {attempt+1}/{tries})")
            _backoff(attempt, hint=2)
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            hint = float(retry_after) if retry_after else None
            last_error[0] = f"HTTP 429 rate limited (server asked {retry_after or '?'}s)"
            print(f"  [classify_batch] {last_error[0]} (attempt {attempt+1}/{tries})")
            _backoff(attempt, hint=hint)
            continue

        if resp.status_code in (500, 502, 503, 504):
            last_error[0] = f"HTTP {resp.status_code}"
            print(f"  [classify_batch] {last_error[0]} (attempt {attempt+1}/{tries})")
            _backoff(attempt)
            continue

        last_error[0] = f"non-retryable HTTP {resp.status_code}: {resp.text[:150]!r}"
        print(f"  [classify_batch] {last_error[0]}")
        return {}

    print(f"  [classify_batch] gave up after {tries} attempts: {last_error[0]}")
    return {}
