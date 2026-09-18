"""Run the LLM classifier over every row in raw_combined.csv that isn't already
in ground_truth_400.json. Checkpoints to a JSONL file after every batch so an
interruption can resume instead of restarting from zero. Rows that fail even
after classify_batch's internal retries get a couple of extra whole-run retry
passes at the end (transient errors are common at this volume)."""
import json
import os
import time

import pandas as pd

import groq_classify as gc

BATCH_SIZE = 6
CHECKPOINT_PATH = "data/llm_classification_progress.jsonl"
EXTRA_RETRY_PASSES = 1
MAX_RUNTIME_SECONDS = 35 * 60  # remaining budget: the 55-min window had ~20 min used before the process was killed; hard cap per user instruction, whatever
                                # the coverage, rather than chase full coverage against an
                                # unpredictable free-tier burst/congestion throttle

df = pd.read_csv("data/raw_combined.csv")
df["text"] = df["text"].fillna("")
df["id"] = df["id"].astype(str)

ground_truth = json.load(open("data/ground_truth_400.json"))
already_tagged_ids = {str(r["id"]) for r in ground_truth}

target = df[~df["id"].isin(already_tagged_ids)].reset_index(drop=True)
print(f"total rows: {len(df)}, already hand-tagged: {len(already_tagged_ids)}, "
      f"to classify: {len(target)}")

done_ids = set()
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"resuming: {len(done_ids)} rows already classified in checkpoint")

remaining = target[~target["id"].isin(done_ids)].reset_index(drop=True)
print(f"remaining to classify this run: {len(remaining)}")

t0 = time.time()
checkpoint_f = open(CHECKPOINT_PATH, "a")
processed = 0
failed_ids = []


def write_record(row, pred):
    record = {
        "id": str(row["id"]), "source": row["source"], "url": row["url"],
        "date": row["date"], "rating": row["rating"], "text": row["text"],
        "is_retrieval_case": bool(pred.get("is_retrieval_case")),
        "cluster": pred.get("cluster"),
        "opportunity_tag": pred.get("opportunity_tag"),
        "photo_type": pred.get("photo_type", ""),
        "what_they_remember": pred.get("what_they_remember", []),
        "what_they_forgot": pred.get("what_they_forgot", []),
        "search_attempt": pred.get("search_attempt", ""),
        "outcome": pred.get("outcome", ""),
        "workaround": pred.get("workaround", ""),
    }
    checkpoint_f.write(json.dumps(record) + "\n")


time_budget_exceeded = False


def run_pass(rows_df, pass_label):
    global processed, time_budget_exceeded
    still_failed = []
    n_batches = (len(rows_df) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        if time.time() - t0 > MAX_RUNTIME_SECONDS:
            time_budget_exceeded = True
            remaining_in_pass = rows_df.iloc[b * BATCH_SIZE:]
            still_failed.extend(remaining_in_pass["id"].tolist())
            print(f"[{pass_label}] time budget ({MAX_RUNTIME_SECONDS}s) reached, stopping "
                  f"with {len(remaining_in_pass)} rows in this pass left unclassified", flush=True)
            break

        chunk = rows_df.iloc[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
        rows = [{"i": idx, "text": row["text"]} for idx, row in chunk.iterrows()]
        result = gc.classify_batch(rows)

        batch_ok = 0
        for idx, row in chunk.iterrows():
            pred = result.get(idx)
            if pred is None:
                still_failed.append(row["id"])
                continue
            write_record(row, pred)
            processed += 1
            batch_ok += 1
        checkpoint_f.flush()
        print(f"[{pass_label}] batch {b+1}/{n_batches} -> {batch_ok}/{len(chunk)} classified, "
              f"{processed} total ({time.time()-t0:.0f}s elapsed)", flush=True)
    return still_failed


failed_ids = run_pass(remaining, "main")

for retry_num in range(1, EXTRA_RETRY_PASSES + 1):
    if not failed_ids or time_budget_exceeded:
        break
    retry_df = target[target["id"].isin(failed_ids)].reset_index(drop=True)
    print(f"\nretry pass {retry_num}: re-attempting {len(retry_df)} failed rows", flush=True)
    failed_ids = run_pass(retry_df, f"retry{retry_num}")

checkpoint_f.close()

if failed_ids:
    with open("data/llm_classification_failures.jsonl", "w") as f:
        for fid in failed_ids:
            f.write(json.dumps({"id": fid}) + "\n")
    print(f"\n{len(failed_ids)} rows not classified this run "
          f"({'time budget reached' if time_budget_exceeded else 'failed after retry passes'}) "
          f"-> data/llm_classification_failures.jsonl")

elapsed = time.time() - t0
print(f"\nDONE. Classified {processed} rows in this run. Wall clock: {elapsed:.1f}s "
      f"({elapsed/60:.1f} min). Time budget exceeded: {time_budget_exceeded}")
