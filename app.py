import json
import os

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
COMBINED_CSV = os.path.join(BASE_DIR, "data", "raw_combined.csv")
FINDINGS_JSON = os.path.join(BASE_DIR, "data", "discovery_findings.json")

st.set_page_config(page_title="Google Photos Retrieval — Discovery Engine", layout="wide")


@st.cache_data
def load_findings():
    with open(FINDINGS_JSON) as f:
        return json.load(f)


@st.cache_data
def load_combined():
    df = pd.read_csv(COMBINED_CSV)
    df["text"] = df["text"].fillna("")
    return df


findings = load_findings()
df = load_combined()

st.title("Google Photos Retrieval — Discovery Engine")
st.caption(
    "Why people can't find an old photo they remember but can't precisely describe or search for."
)

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
            st.metric("Cases", cluster["case_count"])
            for case in cluster["cases"][:3]:
                quote = (case.get("_text") or "").strip().replace("\n", " ")
                if len(quote) > 200:
                    quote = quote[:200].rstrip() + "…"
                url = case.get("_url")
                st.markdown(f"> {quote}")
                if url:
                    st.markdown(f"[source]({url})")
                st.caption(f"{case.get('source', '')} · {case.get('opportunity_tag', '')}")

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
    use_container_width=True,
    height=420,
    column_config={
        "url": st.column_config.LinkColumn("url"),
        "text": st.column_config.TextColumn("text", width="large"),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Methodology
# ---------------------------------------------------------------------------
st.header("Methodology")
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
"""
)
