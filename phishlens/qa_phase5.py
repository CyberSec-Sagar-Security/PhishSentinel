"""Phase 5 QA verification: train/test CSV content check."""
import pandas as pd
from pathlib import Path

train = pd.read_csv("data/processed/train.csv", low_memory=False)
test  = pd.read_csv("data/processed/test.csv",  low_memory=False)

print(f"=== train.csv: {len(train):,} rows ===")
phish_t = int((train["label"]==1).sum())
legit_t = int((train["label"]==0).sum())
print(f"  Phishing : {phish_t:,}")
print(f"  Legit    : {legit_t:,}")
print(f"  Columns  : {list(train.columns)}")
print()
print("Source breakdown (train):")
for src, cnt in train["source"].value_counts().items():
    p = int((train.loc[train["source"]==src,"label"]==1).sum())
    l = int((train.loc[train["source"]==src,"label"]==0).sum())
    print(f"  {src}: {cnt:,} total (phish={p:,}, legit={l:,})")

print()
print(f"=== test.csv: {len(test):,} rows ===")
phish_v = int((test["label"]==1).sum())
legit_v = int((test["label"]==0).sum())
print(f"  Phishing : {phish_v:,}")
print(f"  Legit    : {legit_v:,}")
print()
print("Source breakdown (test):")
for src, cnt in test["source"].value_counts().items():
    p = int((test.loc[test["source"]==src,"label"]==1).sum())
    l = int((test.loc[test["source"]==src,"label"]==0).sum())
    print(f"  {src}: {cnt:,} total (phish={p:,}, legit={l:,})")

print()
print("=== File sizes ===")
for fp in ["data/processed/train.csv","data/processed/test.csv"]:
    sz = Path(fp).stat().st_size
    print(f"  {fp}: {sz/1024/1024:.1f} MB")

print()
print("=== Manifest ===")
import json
m = json.loads(Path("data/processed/dataset_manifest.json").read_text())
print(f"  Generated: {m.get('generated','?')}")
