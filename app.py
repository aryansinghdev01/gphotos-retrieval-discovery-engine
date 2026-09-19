import json
import os
import time

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(__file__)
COMBINED_CSV = os.path.join(BASE_DIR, "data", "raw_combined.csv")
FINDINGS_V2_JSON = os.path.join(BASE_DIR, "data", "discovery_findings_v2.json")
FINDINGS_V1_JSON = os.path.join(BASE_DIR, "data", "discovery_findings.json")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CHAT_MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title="Google Photos Retrieval — Discovery Engine", layout="wide")


def _file_sig(path):
    """Changes whenever the file is replaced, so cached data can't outlive a redeploy."""
    st_ = os.stat(path)
    return (st_.st_mtime_ns, st_.st_size)


@st.cache_data(ttl=300)
def load_findings(sig):
    """Prefer the LLM-scaled v2 findings; fall back to the original 400-row-only
    findings if v2 hasn't been generated yet. Normalizes both into the same shape."""
    if os.path.exists(FINDINGS_V2_JSON):
        with open(FINDINGS_V2_JSON) as f:
            data = json.load(f)
        return {"version": 2, **data}
    with open(FINDINGS_V1_JSON) as f:
        data = json.load(f)
    for c in data["clusters"]:
        for case in c["cases"]:
            case["labeled_by"] = "human"
    return {"version": 1, **data}


@st.cache_data(ttl=300)
def load_combined(sig):
    df = pd.read_csv(COMBINED_CSV)
    df["text"] = df["text"].fillna("")
    df["id"] = df["id"].astype(str)
    return df


@st.cache_resource
def build_retrieval_index(_df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, min_df=2, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(_df["text"].tolist())
    return vectorizer, matrix


def get_groq_api_key():
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY")


def retrieve_rows(question, df, vectorizer, matrix, top_k=8, min_similarity=0.12):
    q_vec = vectorizer.transform([question])
    sims = (matrix @ q_vec.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        if score < min_similarity:
            continue
        row = df.iloc[idx]
        results.append({
            "id": row["id"], "source": row["source"], "text": row["text"],
            "url": row["url"], "similarity": score,
        })
    return results


def ask_groq(question, retrieved, tries=3):
    api_key = get_groq_api_key()
    if not api_key:
        return {"answer": "GROQ_API_KEY isn't configured for this app — the chat can't run without it.",
                "cited_ids": []}

    context = "\n\n".join(
        f'ROW_ID={r["id"]} (source={r["source"]}): {r["text"][:500]}' for r in retrieved
    )
    system_prompt = (
        "You answer questions about a dataset of Google Photos user reviews, posts, and "
        "comments, using ONLY the rows provided below — never outside knowledge. Every claim "
        "must be grounded in these rows.\n\nRows:\n" + context +
        "\n\nOutput STRICT JSON: {\"answer\": \"<your answer, citing ROW_IDs inline like "
        "[id]>\", \"cited_ids\": [\"<row ids you actually used>\"]}. If these rows don't "
        "actually contain information relevant to the question, set answer to a plain "
        "statement that nothing relevant was found and cited_ids to an empty list — do not "
        "guess or use outside knowledge."
    )
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
        "temperature": 0.2,
    }

    for attempt in range(tries):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body, timeout=30,
            )
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 200:
            try:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return {"answer": parsed.get("answer", ""), "cited_ids": parsed.get("cited_ids", [])}
            except (KeyError, ValueError, json.JSONDecodeError):
                return {"answer": "The model returned something I couldn't parse — try rephrasing.",
                        "cited_ids": []}
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after else 10)
            continue
        if resp.status_code in (500, 502, 503, 504):
            time.sleep(3 * (attempt + 1))
            continue
        return {"answer": f"The API returned an error (HTTP {resp.status_code}) — try again shortly.",
                "cited_ids": []}
    return {"answer": "The API is temporarily unavailable — try again in a moment.", "cited_ids": []}


_findings_path = FINDINGS_V2_JSON if os.path.exists(FINDINGS_V2_JSON) else FINDINGS_V1_JSON
findings = load_findings(_file_sig(_findings_path))
df = load_combined(_file_sig(COMBINED_CSV))

st.title("Google Photos Retrieval — Discovery Engine")
st.caption(
    "Why people can't find an old photo they remember but can't precisely describe or search for."
)
if findings["version"] == 2:
    _cov = findings.get("llm_classification_coverage", {})
    st.caption(
        f"Data version: generated {findings.get('generated_at', 'unknown')} · "
        f"{_cov.get('rows_classified', 0):,}/{_cov.get('rows_eligible', 0):,} untagged rows LLM-classified · "
        f"{findings['total_cases']} cases ({findings['human_cases']} human + {findings['llm_cases']} LLM)"
    )
else:
    st.caption("Data version: original hand-tagged findings only (v2 file not found).")

# ---------------------------------------------------------------------------
# 1. Opportunity-area clusters
# ---------------------------------------------------------------------------
st.header("Opportunity areas")

clusters = sorted(findings["clusters"], key=lambda c: c["case_count"], reverse=True)
cols = st.columns(2)

for i, cluster in enumerate(clusters):
    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(cluster["name"])
            metric_cols = st.columns(3) if findings["version"] == 2 else st.columns(1)
            metric_cols[0].metric("Cases", cluster["case_count"])
            if findings["version"] == 2:
                metric_cols[1].metric("Human-labeled", cluster.get("human_count", 0))
                metric_cols[2].metric("LLM-labeled", cluster.get("llm_count", 0))
            humans = [c for c in cluster["cases"] if c.get("labeled_by") == "human"]
            llms = [c for c in cluster["cases"] if c.get("labeled_by") == "llm"]
            for case in (humans[:2] + llms[:1]) or cluster["cases"][:3]:
                quote = (case.get("_text") or "").strip().replace("\n", " ")
                if len(quote) > 200:
                    quote = quote[:200].rstrip() + "…"
                url = case.get("_url")
                st.markdown(f"> {quote}")
                badge = "🧑 human-labeled" if case.get("labeled_by") == "human" else "🤖 LLM-labeled"
                caption = f"{case.get('source', '')} · {case.get('opportunity_tag', '')} · {badge}"
                if url:
                    st.markdown(f"[source]({url})")
                st.caption(caption)

if findings["version"] == 2:
    cov = findings.get("llm_classification_coverage", {})
    st.caption(
        f"Human-labeled = hand-tagged sample ({findings['human_tagged_sample_size']} rows). "
        f"LLM-labeled = {cov.get('rows_classified', 0):,} of {cov.get('rows_eligible', 0):,} remaining rows "
        f"classified by {', '.join('`' + m + '`' for m in findings['classifier']['models_used'])} "
        f"({cov.get('coverage_pct', 0)}% coverage{'' if cov.get('coverage_pct', 0) >= 100 else ' — PARTIAL'}). "
        f"Counts are shown separately, never merged."
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Keyword search over the full raw dataset
# ---------------------------------------------------------------------------
st.header("Search the full dataset")
st.caption(
    f"Verify the findings yourself — plain substring search across all {len(df):,} collected "
    "reviews, posts, and comments (not just the tagged sample)."
)

search_col, filter_col1, filter_col2 = st.columns([3, 1, 1])
with search_col:
    query = st.text_input("Search text (case-insensitive)", value="")
with filter_col1:
    sources = st.multiselect(
        "Source",
        options=sorted(df["source"].unique()),
        default=[],
    )
with filter_col2:
    min_length = st.number_input("Min text length", min_value=0, value=0, step=10)

filtered = df
if query:
    filtered = filtered[filtered["text"].str.contains(query, case=False, na=False, regex=False)]
if sources:
    filtered = filtered[filtered["source"].isin(sources)]
if min_length:
    filtered = filtered[filtered["text"].str.len() >= min_length]

st.write(f"{len(filtered):,} matching rows")
st.dataframe(
    filtered[["text", "source", "rating", "date", "url"]],
    width="stretch",
    height=420,
    column_config={
        "url": st.column_config.LinkColumn("url"),
        "text": st.column_config.TextColumn("text", width="large"),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Ask the data — grounded chat
# ---------------------------------------------------------------------------
st.header("Ask the data")
st.caption(
    "Free-text Q&A grounded in the actual dataset — retrieves the most relevant rows "
    "(TF-IDF over all rows) and has the model answer only from those, citing row IDs. "
    "This is not a general chatbot: if nothing relevant is retrieved, it says so."
)

chat_df = df[df["text"].str.len() >= 40].reset_index(drop=True)
vectorizer, tfidf_matrix = build_retrieval_index(chat_df)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["retrieved"]:
            with st.expander(f"Sources ({len(turn['retrieved'])} rows retrieved)"):
                for r in turn["retrieved"]:
                    cited = "✓ cited" if r["id"] in turn["cited_ids"] else ""
                    st.markdown(
                        f"**[{r['id']}]** {r['source']} · similarity {r['similarity']:.2f} {cited}"
                    )
                    st.caption(r["text"][:250] + ("…" if len(r["text"]) > 250 else ""))
                    if r["url"]:
                        st.markdown(f"[link]({r['url']})")

question = st.chat_input("e.g. what do people say about OCR / text-in-image search?")
if question:
    with st.chat_message("user"):
        st.write(question)
    retrieved = retrieve_rows(question, chat_df, vectorizer, tfidf_matrix)
    with st.chat_message("assistant"):
        if not retrieved:
            answer = "I couldn't find anything relevant to that in the dataset — try rephrasing or asking something more specific to Google Photos retrieval."
            cited_ids = []
            st.write(answer)
        else:
            with st.spinner("Retrieving and answering..."):
                result = ask_groq(question, retrieved)
            answer = result["answer"]
            cited_ids = result["cited_ids"]
            st.write(answer)
            with st.expander(f"Sources ({len(retrieved)} rows retrieved)"):
                for r in retrieved:
                    cited = "✓ cited" if r["id"] in cited_ids else ""
                    st.markdown(f"**[{r['id']}]** {r['source']} · similarity {r['similarity']:.2f} {cited}")
                    st.caption(r["text"][:250] + ("…" if len(r["text"]) > 250 else ""))
                    if r["url"]:
                        st.markdown(f"[link]({r['url']})")
    st.session_state.chat_history.append({
        "question": question, "answer": answer, "retrieved": retrieved, "cited_ids": cited_ids,
    })

st.divider()

# ---------------------------------------------------------------------------
# 4. Methodology
# ---------------------------------------------------------------------------
st.header("Methodology")

if findings["version"] == 2:
    cov = findings.get("llm_classification_coverage", {})
    clf = findings["classifier"]
    _NICE = {"play_store": "Play Store", "app_store": "App Store", "reddit_post": "Reddit post", "reddit_comment": "Reddit comment"}
    _unreached = [_NICE.get(k, k) for k, v in cov.get("coverage_by_source", {}).items() if v["classified"] == 0]
    _unreached_txt = ", ".join(_unreached[:-1]) + (" and " if len(_unreached) > 1 else "") + _unreached[-1] if _unreached else "no"
    _human_by_src = {}
    for _c in findings["clusters"]:
        for _case in _c["cases"]:
            if _case.get("labeled_by") == "human":
                _human_by_src[_case["source"]] = _human_by_src.get(_case["source"], 0) + 1
    _human_src_txt = ", ".join(f"{n} {_NICE.get(s, s)}" for s, n in
                               sorted(_human_by_src.items(), key=lambda kv: -kv[1]))
    st.info(
        f"**Scope of the automated scaling:** it covered Play Store only — {cov.get('rows_classified', 0):,} of "
        f"{cov.get('rows_eligible', 0):,} untagged rows; "
        f"{_unreached_txt} rows were never reached. "
        f"LLMs also over-flag long reviews, so the {findings['llm_cases']} LLM-labeled cases are **not** a representative "
        f"scale-up of the full corpus. The **{findings['human_cases']} hand-verified cases** — from a hand-tagged sample of "
        f"100 rows from each of the four sources ({_human_src_txt}) — are the primary, trustworthy evidence base."
    )
    full = cov.get("rows_classified", 0) >= cov.get("rows_eligible", 1)
    failed = cov.get("rows_failed_after_retries")
    cov_src = "; ".join(f"`{k}` {v['classified']:,}/{v['eligible']:,}" for k, v in cov.get("coverage_by_source", {}).items())
    by_model = ", ".join(f"`{m}`: {n:,} rows" for m, n in clf["rows_classified_by_model"].items())
    st.markdown(
        f"""
- **{findings['total_rows']:,} rows** collected from the Google Play Store, the Apple App Store,
  and Reddit (posts and comments across relevant subreddits and site-wide search).
- **{findings['human_tagged_sample_size']:,} rows** (~{findings['human_tagged_sample_size'] / findings['total_rows'] * 100:.1f}%)
  were hand-tagged against a structured schema distinguishing genuine photo-retrieval failures from
  general complaints — the ground truth. This produced **{findings['human_cases']} human-labeled cases**.
- An LLM classifier (Groq, free tier) was validated blind against those hand-tagged rows before being
  used (numbers below). It picks each case's tag from a **closed set of {clf['closed_tag_set_size']} tags** taken from
  the hand-tagged findings, and the cluster is derived from the tag, so counts roll up into the same
  4 clusters. Tags outside the set are flagged ({clf['llm_cases_with_tag_outside_closed_set']} of {findings['llm_cases']} LLM cases).
- The LLM classified **{cov.get('rows_classified', 0):,} of the {cov.get('rows_eligible', 0):,} untagged rows
  ({cov.get('coverage_pct', 0)}%)**{' — the full remainder of the corpus.' if full else '. **This is PARTIAL coverage** — Groq\'s free tier throttled every model we tried, so the run was stopped to meet a deadline. Treat LLM counts as a lower bound, not a full scan of the corpus.'}
  Rows by classifier: {by_model}. Every case records its `source_model`.
  **Coverage by source: {cov_src}.** The run works through the file in order, so the classified rows are
  {'not a random sample of the corpus — ' + 'they are the Play Store reviews it reached first (short reviews first, then longer ones), and none from the other sources' if any(v['classified'] == 0 and v['eligible'] > 0 for v in cov.get('coverage_by_source', {}).values()) else 'spread across sources'}.
  The two classifiers therefore also saw different kinds of rows (short vs long reviews), so their flag rates are not directly comparable.
  Long reviews were flagged by the LLMs far more often (~43%) than in the hand-tagged sample (17.5%), so the LLM counts likely overstate true cases.
  {(str(clf.get('llm_flagged_but_unplaced', 0)) + ' LLM-flagged rows had no valid cluster and are excluded from the counts. ') if clf.get('llm_flagged_but_unplaced') else ''}{'No rows failed after retries.' if failed == 0 else (str(failed) + ' rows failed even after retries.' if failed else '')}
- That produced **{findings['llm_cases']} LLM-labeled cases**, kept separate from the
  {findings['human_cases']} human-labeled ones in every count above ({findings['total_cases']} total).
"""
    )
    validation_path = os.path.join(BASE_DIR, "data", "validation_report.json")
    if os.path.exists(validation_path):
        with open(validation_path) as f:
            v = json.load(f)
        st.subheader("LLM classifier validation (blind, against 400 hand-tagged rows)")
        vcols = st.columns(4)
        vcols[0].metric("Accuracy", f"{v['accuracy']*100:.1f}%")
        vcols[1].metric("Precision", f"{v['precision']*100:.1f}%")
        vcols[2].metric("Recall", f"{v['recall']*100:.1f}%")
        vcols[3].metric("F1", f"{v['f1']*100:.1f}%")
        st.caption(
            f"Confusion: {v['confusion']['tp']} true positives, {v['confusion']['fp']} false "
            f"positives, {v['confusion']['tn']} true negatives, {v['confusion']['fn']} false "
            f"negatives (is_retrieval_case classification, blind — tags stripped before sending). "
            f"Scored on {v['n_rows'] - v.get('n_missing_predictions', 0)} of {v['n_rows']} rows — "
            f"{v.get('n_missing_predictions', 0)} rows got no prediction (API failures) and are excluded."
        )
        tag_match = v.get("opportunity_tag_exact_match_rate_on_true_positives")
        if tag_match is not None:
            st.caption(
                f"Tag agreement: on the {v['confusion']['tp']} cases both the LLM and the human flagged, the LLM "
                f"chose the same opportunity tag as the human {tag_match*100:.0f}% of the time "
                f"(model: `{v.get('model', 'n/a')}`, closed tag set — every predicted tag came from the fixed list)."
            )
    mini_path = os.path.join(BASE_DIR, "data", "validation_report_mini.json")
    if os.path.exists(mini_path):
        with open(mini_path) as f:
            m = json.load(f)
        st.subheader(f"Second classifier: `{m['model']}` (mini-validation, {m['sample']['n']} rows)")
        mc = m["this_model"]
        mcols = st.columns(3)
        mcols[0].metric("Recall", f"{mc['recall']*100:.0f}%")
        mcols[1].metric("Precision (at real 12.5% base rate)", f"{(mc['precision_at_real_prevalence'] or 0)*100:.0f}%")
        mcols[2].metric("Agreement w/ gpt-oss-120b", f"{(m['agreement_with_gpt_oss_120b_on_is_retrieval_case'] or 0)*100:.0f}%")
        st.caption(
            f"Small sample ({m['sample']['positives']} positives + {m['sample']['negatives']} negatives, "
            f"positives oversampled) — treat these as rough, not tight, estimates. "
            f"{m['rows_predicted']} of {m['sample']['n']} rows got a prediction."
        )
else:
    st.markdown(
        f"""
- **{findings['total_rows']:,} rows** collected from the Google Play Store, the Apple App Store,
  and Reddit (posts and comments across relevant subreddits and site-wide search).
- **{findings['sample_size']:,} rows** (~{findings['sample_size'] / findings['total_rows'] * 100:.1f}%)
  were manually tagged against a structured schema distinguishing genuine photo-retrieval failures
  ("I remember this photo exists but can't find it") from general complaints, bugs, and unrelated
  feedback.
- **{sum(c['case_count'] for c in clusters)} confirmed retrieval-failure cases** were found in the
  tagged sample — a **{findings['hit_rate'] * 100:.1f}% hit rate**, consistent across two
  independent sample draws.
- *(LLM-scaled classification across the full corpus hasn't been merged in yet — this is the
  original hand-tagged-sample-only view.)*
"""
    )
