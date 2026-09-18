"""Merge the 50 human-verified retrieval cases (from ground_truth_400.json) with
the LLM-classified cases (from the checkpoint file) into a single full-corpus
findings file, keeping provenance ("human-verified" vs "llm-classified") on
every case so the deck can show both."""
import json

CLUSTER_NAMES = {
    1: "Browsing/visual structure removed or broken",
    2: "Search itself is broken",
    3: "Content exists but isn't indexed where expected",
    4: "Fragmentation / apparent loss",
}

ground_truth = json.load(open("data/ground_truth_400.json"))
old_findings = json.load(open("data/discovery_findings.json"))

# map opportunity_tag -> cluster name, learned from the original human clustering
tag_to_cluster = {}
for cluster in old_findings["clusters"]:
    for case in cluster["cases"]:
        tag_to_cluster[case["opportunity_tag"]] = cluster["name"]

clusters = {name: [] for name in CLUSTER_NAMES.values()}

for row in ground_truth:
    if not row.get("is_retrieval_case"):
        continue
    tag = row.get("opportunity_tag")
    cluster_name = tag_to_cluster.get(tag)
    if cluster_name is None:
        # fall back: shouldn't happen since these are the original 50, but stay safe
        cluster_name = "Content exists but isn't indexed where expected"
    clusters[cluster_name].append({
        "verification": "human-verified",
        "source": row["source"], "id": row["id"], "opportunity_tag": tag,
        "photo_type": row.get("photo_type", ""),
        "what_they_remember": row.get("what_they_remember", []),
        "what_they_forgot": row.get("what_they_forgot", []),
        "search_attempt": row.get("search_attempt", ""),
        "outcome": row.get("outcome", ""),
        "workaround": row.get("workaround", ""),
        "_text": row["text"], "_url": row["url"],
    })

n_llm_true = 0
n_llm_total = 0
try:
    with open("data/llm_classification_progress.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_llm_total += 1
            if not row.get("is_retrieval_case"):
                continue
            n_llm_true += 1
            try:
                cluster_num = int(row.get("cluster"))
            except (TypeError, ValueError):
                cluster_num = None
            cluster_name = CLUSTER_NAMES.get(cluster_num)
            if cluster_name is None:
                continue
            clusters[cluster_name].append({
                "verification": "llm-classified",
                "source": row["source"], "id": row["id"],
                "opportunity_tag": row.get("opportunity_tag"),
                "photo_type": row.get("photo_type", ""),
                "what_they_remember": row.get("what_they_remember", []),
                "what_they_forgot": row.get("what_they_forgot", []),
                "search_attempt": row.get("search_attempt", ""),
                "outcome": row.get("outcome", ""),
                "workaround": row.get("workaround", ""),
                "_text": row["text"], "_url": row["url"],
            })
except FileNotFoundError:
    pass

out_clusters = []
for name, cases in clusters.items():
    out_clusters.append({
        "name": name,
        "case_count": len(cases),
        "human_verified_count": sum(1 for c in cases if c["verification"] == "human-verified"),
        "llm_classified_partial_count": sum(1 for c in cases if c["verification"] == "llm-classified"),
        "cases": cases,
    })
out_clusters.sort(key=lambda c: c["case_count"], reverse=True)

total_rows = int(json.load(open("data/discovery_findings.json"))["total_rows"])
rows_eligible_for_llm = total_rows - len(ground_truth)
total_cases = sum(c["case_count"] for c in out_clusters)

report = {
    "total_rows": total_rows,
    "human_tagged_sample_size": len(ground_truth),
    "human_verified_cases": sum(c["human_verified_count"] for c in out_clusters),
    "llm_classification_coverage": {
        "rows_eligible": rows_eligible_for_llm,
        "rows_classified": n_llm_total,
        "coverage_pct": round(n_llm_total / rows_eligible_for_llm * 100, 1) if rows_eligible_for_llm else 0,
        "note": (
            f"LLM classification of the full corpus was capped at a bounded wall-clock time "
            f"budget due to unpredictable burst/congestion throttling on Groq's free tier. "
            f"Only {n_llm_total:,} of {rows_eligible_for_llm:,} remaining (not hand-tagged) rows "
            f"were classified in this run — this is PARTIAL coverage, not the full corpus."
        ),
    },
    "llm_classified_partial_cases": n_llm_true,
    "total_confirmed_cases": total_cases,
    "clusters": out_clusters,
}

with open("data/discovery_findings_v2.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"wrote data/discovery_findings_v2.json: {total_cases} total cases "
      f"({report['human_verified_cases']} human-verified + {report['llm_classified_partial_cases']} "
      f"llm-classified-partial) across {len(out_clusters)} clusters")
print(f"LLM coverage: {n_llm_total:,}/{rows_eligible_for_llm:,} rows "
      f"({report['llm_classification_coverage']['coverage_pct']}%)")
for c in out_clusters:
    print(f"  {c['name']}: {c['case_count']} (human {c['human_verified_count']} / "
          f"llm-partial {c['llm_classified_partial_count']})")
