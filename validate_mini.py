"""Mini blind validation on a fixed 100-row sample of the hand-tagged rows
(30 positives + 70 negatives, seed 42), compared against gpt-oss-120b on the SAME rows.

Positives are oversampled so recall means something with only 100 rows, which inflates raw
precision relative to the real ~12.5% base rate; a prevalence-adjusted precision is reported too.
Has a hard wall-clock deadline: if the model is throttled, it stops and says so.
"""
import json
import os
import random
import time

import groq_classify as gc

N_POS, N_NEG, SEED = 30, 70, 42
BATCH_SIZE = 10
DEADLINE = float(os.environ.get("MINI_DEADLINE_SECONDS", "600"))
REAL_PREVALENCE = 0.125
OUT = os.environ.get("VALIDATION_REPORT_PATH", "data/validation_report_mini.json")

gt = json.load(open("data/ground_truth_400.json"))
rng = random.Random(SEED)
pos = [i for i, r in enumerate(gt) if r["is_retrieval_case"]]
neg = [i for i, r in enumerate(gt) if not r["is_retrieval_case"]]
sample = sorted(rng.sample(pos, N_POS) + rng.sample(neg, N_NEG))

t0 = time.time()
preds, pending, throttled = {}, list(sample), False
for attempt in range(4):
    if not pending or throttled:
        break
    still = []
    for b in range(0, len(pending), BATCH_SIZE):
        idxs = pending[b:b + BATCH_SIZE]
        if time.time() - t0 > DEADLINE:
            throttled = True
            still.extend(pending[b:])
            print(f"DEADLINE ({DEADLINE:.0f}s) reached — treating as heavy throttling", flush=True)
            break
        res = gc.classify_batch([{"i": i, "text": gt[i]["text"]} for i in idxs], tries=3)
        preds.update(res)
        still.extend(i for i in idxs if i not in res)
        print(f"[pass {attempt}] {len(preds)}/{len(sample)} rows predicted | {time.time()-t0:.0f}s", flush=True)
    pending = still


def metrics(pairs):
    tp = sum(1 for a, p in pairs if a and p)
    fp = sum(1 for a, p in pairs if not a and p)
    tn = sum(1 for a, p in pairs if not a and not p)
    fn = sum(1 for a, p in pairs if a and not p)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    denom = rec * REAL_PREVALENCE + fpr * (1 - REAL_PREVALENCE)
    return {"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}, "precision_raw": round(prec, 4),
            "recall": round(rec, 4), "f1_raw": round(f1, 4),
            "precision_at_real_prevalence": round(rec * REAL_PREVALENCE / denom, 4) if denom else None,
            "false_positive_rate": round(fpr, 4)}


scored = [i for i in sample if i in preds]
new = metrics([(gt[i]["is_retrieval_case"], bool(preds[i]["is_retrieval_case"])) for i in scored])

# gpt-oss-120b on the same rows, from its full validation report
ref = {r["i"]: r for r in json.load(open("data/validation_report_gpt-oss-120b.json"))["per_row"]}
ref_pairs = [(gt[i]["is_retrieval_case"], bool(ref[i]["predicted"])) for i in scored if ref.get(i, {}).get("predicted") is not None]
tp_rows = [i for i in scored if gt[i]["is_retrieval_case"] and preds[i]["is_retrieval_case"]]
report = {
    "model": gc.MODEL,
    "sample": {"n": len(sample), "positives": N_POS, "negatives": N_NEG, "seed": SEED},
    "rows_predicted": len(scored), "rows_missing": len(sample) - len(scored),
    "hit_deadline_treated_as_throttled": throttled,
    "elapsed_seconds": round(time.time() - t0, 1),
    "this_model": new,
    "gpt_oss_120b_same_rows": metrics(ref_pairs),
    "opportunity_tag_exact_match_on_true_positives":
        round(sum(1 for i in tp_rows if preds[i].get("opportunity_tag") == gt[i].get("opportunity_tag")) / len(tp_rows), 4) if tp_rows else None,
    "tag_in_closed_set_rate_on_true_positives":
        round(sum(1 for i in tp_rows if preds[i].get("tag_in_closed_set")) / len(tp_rows), 4) if tp_rows else None,
    "agreement_with_gpt_oss_120b_on_is_retrieval_case":
        round(sum(1 for i in scored if ref.get(i, {}).get("predicted") == bool(preds[i]["is_retrieval_case"])) / len(scored), 4) if scored else None,
}
json.dump(report, open(OUT, "w"), indent=2)
print("\n=== MINI VALIDATION REPORT ===")
print(json.dumps(report, indent=2))
