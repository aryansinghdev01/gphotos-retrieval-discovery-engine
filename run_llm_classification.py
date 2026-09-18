"""Classify every row in raw_combined.csv that isn't in ground_truth_400.json.

- Checkpoints to a JSONL file after every batch; re-running resumes where it left off.
- Rows the model omitted or that failed are retried in repeated passes (each call also has
  its own exponential backoff); anything still failing is logged with a reason, never
  silently dropped.
- Set MAX_RUNTIME_SECONDS in the environment to cap a run (unset = no cap).
"""
import json
import os
import time

import pandas as pd

import groq_classify as gc

BATCH_SIZE = 10
CHECKPOINT_PATH = "data/llm_classification_v2.jsonl"
FAILURES_PATH = "data/llm_classification_failures.jsonl"
SUMMARY_PATH = "data/llm_run_summary.json"
RETRY_PASSES = 6
MAX_RUNTIME_SECONDS = float(os.environ["MAX_RUNTIME_SECONDS"]) if os.environ.get("MAX_RUNTIME_SECONDS") else None

df = pd.read_csv("data/raw_combined.csv")
df["text"] = df["text"].fillna("")
df["id"] = df["id"].astype(str)

ground_truth = json.load(open("data/ground_truth_400.json"))
already_tagged_ids = {str(r["id"]) for r in ground_truth}
target = df[~df["id"].isin(already_tagged_ids)].reset_index(drop=True)


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


done_ids = {r["id"] for r in read_jsonl(CHECKPOINT_PATH)}
remaining = target[~target["id"].isin(done_ids)].reset_index(drop=True)
print(f"total rows: {len(df)} | hand-tagged: {len(already_tagged_ids)} | to classify: {len(target)} | "
      f"already done: {len(done_ids)} | remaining this run: {len(remaining)}", flush=True)

t0 = time.time()
checkpoint_f = open(CHECKPOINT_PATH, "a")
processed = 0
failure_reasons = {}  # id -> last reason


def out_of_time():
    return MAX_RUNTIME_SECONDS is not None and time.time() - t0 > MAX_RUNTIME_SECONDS


def write_record(row, pred):
    checkpoint_f.write(json.dumps({
        "id": str(row["id"]), "source": row["source"], "url": row["url"],
        "date": row["date"], "rating": row["rating"], "text": row["text"],
        "labeled_by": "llm",
        "source_model": gc.MODEL,
        "is_retrieval_case": bool(pred.get("is_retrieval_case")),
        "cluster": pred.get("cluster"),
        "opportunity_tag": pred.get("opportunity_tag"),
        "tag_in_closed_set": pred.get("tag_in_closed_set"),
        "photo_type": pred.get("photo_type", ""),
        "what_they_remember": pred.get("what_they_remember", []),
        "what_they_forgot": pred.get("what_they_forgot", []),
        "search_attempt": pred.get("search_attempt", ""),
        "outcome": pred.get("outcome", ""),
        "workaround": pred.get("workaround", ""),
    }) + "\n")


def run_pass(rows_df, label):
    """Returns ids still unclassified after this pass."""
    global processed
    left = []
    n_batches = (len(rows_df) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        chunk = rows_df.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        if out_of_time():
            left.extend(rows_df.iloc[b * BATCH_SIZE:]["id"].tolist())
            print(f"[{label}] time cap reached; {len(left)} rows left in this pass", flush=True)
            break
        result = gc.classify_batch([{"i": idx, "text": row["text"]} for idx, row in chunk.iterrows()])
        ok = 0
        for idx, row in chunk.iterrows():
            pred = result.get(idx)
            if pred is None:
                left.append(row["id"])
                failure_reasons[row["id"]] = gc.last_error[0] or "model omitted this row from its response"
                continue
            write_record(row, pred)
            failure_reasons.pop(row["id"], None)
            processed += 1
            ok += 1
        checkpoint_f.flush()
        print(f"[{label}] batch {b+1}/{n_batches}: {ok}/{len(chunk)} ok | {processed} this run | "
              f"{time.time()-t0:.0f}s elapsed", flush=True)
    return left


pending_ids = run_pass(remaining, "main")
for n in range(1, RETRY_PASSES + 1):
    if not pending_ids or out_of_time():
        break
    print(f"\nretry pass {n}/{RETRY_PASSES}: {len(pending_ids)} rows still unclassified", flush=True)
    pending_ids = run_pass(target[target["id"].isin(pending_ids)].reset_index(drop=True), f"retry{n}")

checkpoint_f.close()

with open(FAILURES_PATH, "w") as f:
    for rid in pending_ids:
        f.write(json.dumps({"id": rid, "reason": failure_reasons.get(rid, "not attempted (time cap)")}) + "\n")

elapsed = time.time() - t0
prev = json.load(open(SUMMARY_PATH)) if os.path.exists(SUMMARY_PATH) else {"total_elapsed_seconds": 0, "runs": 0}
summary = {
    "total_elapsed_seconds": round(prev["total_elapsed_seconds"] + elapsed, 1),
    "runs": prev["runs"] + 1,
    "last_run_elapsed_seconds": round(elapsed, 1),
    "last_run_classified": processed,
    "rows_classified_total": len(read_jsonl(CHECKPOINT_PATH)),
    "rows_eligible": len(target),
    "rows_failed_after_retries": len(pending_ids),
    "time_cap_hit": out_of_time(),
}
json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
print(f"\nDONE. {json.dumps(summary)}")
