"""Blind validation pass: classify the 400 hand-tagged ground-truth rows with the
LLM (tags stripped before sending) and compare against the human labels."""
import json
import os
import time

import groq_classify as gc

BATCH_SIZE = 10
RETRY_PASSES = 6

ground_truth = json.load(open("data/ground_truth_400.json"))
by_index = {i: row for i, row in enumerate(ground_truth)}

t0 = time.time()
predictions = {}
pending = list(range(len(ground_truth)))
for attempt in range(1 + RETRY_PASSES):
    if not pending:
        break
    label = "main" if attempt == 0 else f"retry{attempt}"
    if attempt:
        print(f"\nretry pass {attempt}: {len(pending)} rows without a prediction", flush=True)
    still = []
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        idxs = pending[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        result = gc.classify_batch([{"i": i, "text": ground_truth[i]["text"]} for i in idxs])
        predictions.update(result)
        still.extend(i for i in idxs if i not in result)
        print(f"[{label}] batch {b+1}/{n_batches} -> {len(result)}/{len(idxs)} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    pending = still

elapsed = time.time() - t0
print(f"\nvalidation classification took {elapsed:.1f}s for {len(ground_truth)} rows")

tp = fp = tn = fn = missing = 0
tag_matches = 0
tag_comparable = 0
per_row = []
for i, row in by_index.items():
    pred = predictions.get(i)
    actual = row["is_retrieval_case"]
    if pred is None:
        missing += 1
        per_row.append({"i": i, "id": row["id"], "actual": actual, "predicted": None})
        continue
    predicted = bool(pred.get("is_retrieval_case"))
    if actual and predicted:
        tp += 1
    elif not actual and predicted:
        fp += 1
    elif not actual and not predicted:
        tn += 1
    elif actual and not predicted:
        fn += 1

    if actual and predicted:
        tag_comparable += 1
        if pred.get("opportunity_tag") == row.get("opportunity_tag"):
            tag_matches += 1

    per_row.append({
        "i": i, "id": row["id"], "actual": actual, "predicted": predicted,
        "actual_tag": row.get("opportunity_tag"), "predicted_tag": pred.get("opportunity_tag"),
    })

total_scored = tp + fp + tn + fn
accuracy = (tp + tn) / total_scored if total_scored else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

report = {
    "n_rows": len(ground_truth),
    "n_missing_predictions": missing,
    "retry_passes_used": attempt,
    "elapsed_seconds": round(elapsed, 1),
    "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "predicted_tag_in_closed_set_rate_on_true_positives": round(
        sum(1 for r in per_row if r.get("actual") and r.get("predicted") and predictions[r["i"]].get("tag_in_closed_set")) /
        max(1, tp), 4),
    "opportunity_tag_exact_match_rate_on_true_positives": round(tag_matches / tag_comparable, 4) if tag_comparable else None,
    "per_row": per_row,
}

report["model"] = gc.MODEL
with open(os.environ.get("VALIDATION_REPORT_PATH", "data/validation_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n=== VALIDATION REPORT ===")
print(json.dumps({k: v for k, v in report.items() if k != "per_row"}, indent=2))
