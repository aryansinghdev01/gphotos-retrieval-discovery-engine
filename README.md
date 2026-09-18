# Google Photos retrieval research — data collection

Free, runs entirely on your own machine in VS Code. No API keys for this part.

## Setup (one time)

1. Open this folder in VS Code.
2. Open a terminal in VS Code (`` Ctrl+` ``) and run:
   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Run it

```
python collect_reviews.py
```

This will take a few minutes (Reddit collection is deliberately rate-limited so it
doesn't get blocked). When it finishes you'll have, inside `data/`:

- `raw_play_store.csv`
- `raw_app_store.csv`
- `raw_reddit.csv`
- `raw_combined.csv` — all three merged, this is the one that matters

## Next step (also free)

Upload `data/raw_combined.csv` into your Claude chat and ask for it to be tagged
and clustered into opportunity areas. That analysis step runs on your existing
Claude access — no separate paid API key needed.

## If something breaks

- **App Store scrape fails / returns 0 rows**: Apple rate-limits this pretty
  aggressively. The script continues without it — Play Store + Reddit alone is
  still a solid dataset.
- **Reddit returns very few results**: try re-running later, or widen
  `SEARCH_TERMS` / `SUBREDDITS` in `collect_reviews.py`.
- **`ModuleNotFoundError`**: make sure the virtual environment is activated
  (you should see `(venv)` in your terminal prompt) before running `pip install`.
