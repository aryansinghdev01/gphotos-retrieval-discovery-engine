"""Merge the 50 human-verified retrieval cases (ground_truth_400.json) with the
LLM-classified cases (llm_classification_v2.jsonl) into discovery_findings_v2.json.

Every case carries labeled_by: "human" | "llm" ("source" is already the platform:
play_store / app_store / reddit_*). Counts are kept separate per cluster and are
never silently summed into one unlabeled number."""
import json
import os

import groq_classify as gc

CLUSTER_NAMES = gc.CLUSTER_NAMES
CHECKPOINT = "data/llm_classification_v2.jsonl"


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


ground_truth = json.load(open("data/ground_truth_400.json"))
old_findings = json.load(open("data/discovery_findings.json"))
tag_to_cluster_name = {}
for cluster in old_findings["clusters"]:
    for case in cluster["cases"]:
        tag_to_cluster_name[case["opportunity_tag"]] = cluster["name"]

clusters = {name: [] for name in CLUSTER_NAMES.values()}


def case_dict(row, labeled_by, text_key, url_key, extra=None):
    d = {
        "labeled_by": labeled_by,
        "source_model": "human" if labeled_by == "human" else None,
        "source": row["source"], "id": str(row["id"]), "opportunity_tag": row.get("opportunity_tag"),
        "photo_type": row.get("photo_type", ""),
        "what_they_remember": row.get("what_they_remember", []),
        "what_they_forgot": row.get("what_they_forgot", []),
        "search_attempt": row.get("search_attempt", ""),
        "outcome": row.get("outcome", ""),
        "workaround": row.get("workaround", ""),
        "_text": row[text_key], "_url": row[url_key],
    }
    d.update(extra or {})
    return d


for row in ground_truth:
    if row.get("is_retrieval_case"):
        name = tag_to_cluster_name[row["opportunity_tag"]]
        clusters[name].append(case_dict(row, "human", "text", "url"))

llm_rows = read_jsonl(CHECKPOINT)
llm_positive = 0
llm_unplaced = []
other_tag_cases = 0
for row in llm_rows:
    if not row.get("is_retrieval_case"):
        continue
    llm_positive += 1
    try:
        cluster_name = CLUSTER_NAMES.get(int(row.get("cluster")))
    except (TypeError, ValueError):
        cluster_name = None
    if cluster_name is None:
        llm_unplaced.append(row["id"])
        continue
    in_set = bool(row.get("tag_in_closed_set"))
    other_tag_cases += 0 if in_set else 1
    clusters[cluster_name].append(case_dict(row, "llm", "text", "url",
                                            {"tag_in_closed_set": in_set, "source_model": row.get("source_model")}))

out_clusters = []
for name, cases in clusters.items():
    human = sum(1 for c in cases if c["labeled_by"] == "human")
    llm = sum(1 for c in cases if c["labeled_by"] == "llm")
    out_clusters.append({"name": name, "case_count": human + llm,
                         "human_count": human, "llm_count": llm, "cases": cases})
out_clusters.sort(key=lambda c: c["case_count"], reverse=True)

total_rows = int(old_findings["total_rows"])
eligible = total_rows - len(ground_truth)
classified = len({r["id"] for r in llm_rows})
classified_ids = {r["id"] for r in llm_rows}
FAIL_PATH = "data/llm_classification_failures.jsonl"
# The failure log is rewritten when a run finishes, so it is only trustworthy if it is newer
# than the checkpoint; otherwise a run is in progress / was interrupted and it is stale.
failures_current = os.path.exists(FAIL_PATH) and os.path.getmtime(FAIL_PATH) >= os.path.getmtime(CHECKPOINT)
failures = [f for f in read_jsonl(FAIL_PATH) if f["id"] not in classified_ids] if failures_current else []
summary = json.load(open("data/llm_run_summary.json")) if os.path.exists("data/llm_run_summary.json") else {}
validation = json.load(open("data/validation_report.json"))
from collections import Counter
rows_by_model = Counter(r.get("source_model") for r in llm_rows)
cases_by_model = Counter(c["source_model"] for cl in out_clusters for c in cl["cases"] if c["labeled_by"] == "llm")
human_cases = sum(c["human_count"] for c in out_clusters)
llm_cases = sum(c["llm_count"] for c in out_clusters)
coverage_pct = round(classified / eligible * 100, 1) if eligible else 0

report = {
    "total_rows": total_rows,
    "human_tagged_sample_size": len(ground_truth),
    "human_cases": human_cases,
    "llm_cases": llm_cases,
    "total_cases": human_cases + llm_cases,
    "classifier": {
        "models_used": sorted(m for m in rows_by_model if m),
        "rows_classified_by_model": dict(rows_by_model),
        "llm_cases_by_model": dict(cases_by_model),
        "closed_tag_set_size": len(gc.TAG_TO_CLUSTER),
        "llm_cases_with_tag_outside_closed_set": other_tag_cases,
        "llm_flagged_but_unplaced": len(llm_unplaced),
    },
    "llm_classification_coverage": {
        "rows_eligible": eligible,
        "rows_classified": classified,
        "coverage_pct": coverage_pct,
        "rows_not_yet_classified": eligible - classified,
        "rows_failed_after_retries": len(failures) if failures_current else None,
        "total_elapsed_seconds": summary.get("total_elapsed_seconds"),
        "note": (
            f"All {eligible:,} untagged rows were classified by the LLM."
            if classified >= eligible else
            f"PARTIAL coverage: only {classified:,} of {eligible:,} untagged rows were classified by the LLM."
        ),
    },
    "clusters": out_clusters,
}

with open("data/discovery_findings_v2.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"wrote data/discovery_findings_v2.json | models={report['classifier']['rows_classified_by_model']}")
print(f"LLM coverage: {classified:,}/{eligible:,} ({coverage_pct}%) | not yet classified: {eligible - classified} | "
      f"failed after retries: {len(failures) if failures_current else 'n/a (run unfinished)'} | "
      f"elapsed: {summary.get('total_elapsed_seconds')}s")
print(f"cases: {human_cases} human + {llm_cases} llm = {human_cases + llm_cases} "
      f"(llm positives={llm_positive}, unplaced={len(llm_unplaced)}, outside closed tag set={other_tag_cases})")
print(f"{'cluster':55s} human   llm  total")
for c in out_clusters:
    print(f"{c['name']:55s} {c['human_count']:5d} {c['llm_count']:5d} {c['case_count']:6d}")
if llm_unplaced:
    print(f"WARNING: {len(llm_unplaced)} LLM-flagged rows had no valid cluster and are NOT in any cluster count")
