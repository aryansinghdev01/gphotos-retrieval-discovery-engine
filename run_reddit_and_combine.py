"""One-off runner: reuse the already-collected Play Store / App Store CSVs
(no need to re-scrape those, they're done and saved) and only run the
Reddit collection + final combine step."""
import os
import pandas as pd
import collect_reviews as cr

df_play = pd.read_csv(os.path.join(cr.OUT_DIR, "raw_play_store.csv"))
df_app = pd.read_csv(os.path.join(cr.OUT_DIR, "raw_app_store.csv"))
print(f"[reused] play store: {len(df_play)} rows, app store: {len(df_app)} rows")

df_reddit = cr.collect_reddit()

df_all = pd.concat([df_play, df_app, df_reddit], ignore_index=True)
df_all = df_all[df_all["text"].str.len() > 15].reset_index(drop=True)
df_all.to_csv(os.path.join(cr.OUT_DIR, "raw_combined.csv"), index=False)

print(f"\nTotal usable items: {len(df_all)}")
print(df_all["source"].value_counts())
print("\nAll set. Upload data/raw_combined.csv to your Claude chat next for the tagging/analysis step.")
