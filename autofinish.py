"""Watch the classification run; when it finishes (or the stop rule fires), merge + deploy.

Stop rule (something actually broke, not just normal 429 retries):
  * the run process died without printing its DONE marker -> restart once, then stop
  * the checkpoint has not grown for STALL_MINUTES while the process is alive -> stop it
On a stop, whatever is classified so far is merged and deployed, labeled PARTIAL.

Usage: autofinish.py [--dry-run]   (dry run: merge + report only, no git, no waiting)
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

os.chdir("/Users/aryansingh/discovery-engine")
PY = "venv/bin/python3"
CK = "data/llm_classification_v2.jsonl"
STATUS = "data/final_status.json"
STALL_MINUTES = 30
POLL_SECONDS = 60
MAX_RESTARTS = 1
DRY = "--dry-run" in sys.argv
LOGS = ["run_llm_log6.txt"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def run_alive():
    return subprocess.run(["pgrep", "-f", "run_llm_classification.py"], capture_output=True).stdout.strip() != b""


def n_rows():
    return sum(1 for line in open(CK) if line.strip()) if os.path.exists(CK) else 0


def run_done():
    p = LOGS[-1]
    return os.path.exists(p) and "\nDONE." in ("\n" + open(p).read())


def start_run():
    p = f"run_llm_log{6 + len(LOGS)}.txt"
    LOGS.append(p)
    env = dict(os.environ, GROQ_MODEL="qwen/qwen3.8-27b", GROQ_MIN_INTERVAL="3")
    subprocess.Popen(["caffeinate", "-i", PY, "-u", "run_llm_classification.py"],
                     stdout=open(p, "w"), stderr=subprocess.STDOUT, env=env)
    log(f"(re)started classification run -> {p}")


def wait_for_end():
    last_n, last_growth, restarts = n_rows(), time.time(), 0
    while True:
        time.sleep(POLL_SECONDS)
        n = n_rows()
        if n > last_n:
            last_n, last_growth = n, time.time()
        alive = run_alive()
        if not alive and run_done():
            return "complete", n
        if not alive:
            if restarts < MAX_RESTARTS:
                restarts += 1
                log("run process died without finishing; restarting once (resumable)")
                start_run()
                last_growth = time.time()
                continue
            return f"stopped: run crashed twice without finishing ({n} rows classified)", n
        if time.time() - last_growth > STALL_MINUTES * 60:
            subprocess.run(["pkill", "-f", "run_llm_classification.py"])
            time.sleep(3)
            return f"stopped: no new rows for {STALL_MINUTES} minutes ({n} rows classified)", n


def finalize(reason, n):
    status = {"reason": reason, "finalized_at": now(), "rows_in_checkpoint": n, "dry_run": DRY}
    r = subprocess.run([PY, "merge_findings.py"], capture_output=True, text=True)
    status["merge_ok"] = r.returncode == 0
    status["merge_output"] = r.stdout[-1500:] + r.stderr[-500:]
    log("merge finished (ok=%s)" % status["merge_ok"])
    if not status["merge_ok"]:
        json.dump(status, open(STATUS, "w"), indent=2)
        return

    d = json.load(open("data/discovery_findings_v2.json"))
    if any("qwen" in m for m in d["classifier"]["models_used"]):
        shutil.copy("data/validation_report_qwen3.8-27b_mini.json", "data/validation_report_mini.json")
    status["per_cluster"] = {c["name"]: {"human": c["human_count"], "llm": c["llm_count"], "total": c["case_count"]}
                             for c in d["clusters"]}
    status["coverage"] = d["llm_classification_coverage"]
    status["rows_by_model"] = d["classifier"]["rows_classified_by_model"]
    status["llm_cases_by_model"] = d["classifier"]["llm_cases_by_model"]
    status["totals"] = {"human": d["human_cases"], "llm": d["llm_cases"], "total": d["total_cases"]}

    if DRY:
        json.dump(status, open(STATUS, "w"), indent=2)
        log("dry run: skipping git")
        return

    files = ["merge_findings.py", "app.py", "data/discovery_findings_v2.json", "data/validation_report_mini.json"]
    g = ["git", "-c", "user.email=aryansingh.dev01@gmail.com", "-c", "user.name=aryansingh"]
    subprocess.run(["git", "add"] + files)
    cov = d["llm_classification_coverage"]
    msg = (f"Update findings: {cov['rows_classified']:,}/{cov['rows_eligible']:,} untagged rows LLM-classified "
           f"({cov['coverage_pct']}%), {d['human_cases']} human + {d['llm_cases']} LLM cases\n\n"
           f"Run outcome: {reason}\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>")
    c = subprocess.run(g + ["commit", "-m", msg], capture_output=True, text=True)
    status["commit_output"] = (c.stdout + c.stderr)[-300:]
    pushed = False
    for attempt in range(4):
        p = subprocess.run(["git", "push", "origin", "master"], capture_output=True, text=True)
        status["push_output"] = (p.stdout + p.stderr)[-300:]
        if p.returncode == 0:
            pushed = True
            break
        time.sleep(30)
    status["pushed"] = pushed
    status["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"pushed={pushed} commit={status['commit']}")

    time.sleep(120)  # give Streamlit Cloud time to pick up the push
    try:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0"
        resp = s.get("https://gphotos-retrieval-discovery-engine-uetv3fw23bkipk3y6ntanc.streamlit.app/", timeout=60)
        status["live_app_http"] = resp.status_code
    except Exception as e:
        status["live_app_http"] = f"error: {type(e).__name__}"
    json.dump(status, open(STATUS, "w"), indent=2)
    log(f"live app http={status['live_app_http']}; wrote {STATUS}")


if DRY:
    finalize("dry run (not waiting for the run)", n_rows())
    print(open(STATUS).read())
else:
    log(f"watching run (stall limit {STALL_MINUTES} min, {n_rows()} rows so far)")
    reason, n = wait_for_end()
    log(f"run ended: {reason}")
    finalize(reason, n)
