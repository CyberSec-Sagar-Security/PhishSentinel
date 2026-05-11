"""Phase 7 sanity check — data loading isolation test."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

print("=== Sanity Check 3: Data Loading Isolation Test ===")

# 1. Existing processed CSV
train_csv = "data/processed/train.csv"
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv, low_memory=False)
    print(f"\nExisting train.csv: {len(df):,} rows")
    print(f"  Phishing : {(df['label']==1).sum():,}")
    print(f"  Legitimate: {(df['label']==0).sum():,}")
    print(f"  Sources  : {df['source'].value_counts().to_dict()}")
else:
    print("No existing train.csv — will be rebuilt by download_datasets.py")

# 2. Quick loader test on LLMGen (smallest HF dataset, ~6.8k rows)
print("\n--- LLMGen loader test (6.8k rows) ---")
from src.ingestion.dataset_loader import load_hf_llmgen
df_llm = load_hf_llmgen()
print(f"LLMGen rows : {len(df_llm):,}")
print(f"Columns     : {list(df_llm.columns)}")
print(f"Labels      : {df_llm['label'].value_counts().to_dict()}")
print(f"Source      : {df_llm['source'].unique()}")
assert (df_llm['label'] == 1).sum() > 0, "FAIL: no phishing rows in LLMGen"
assert (df_llm['label'] == 0).sum() > 0, "FAIL: no legitimate rows in LLMGen"
print("LLMGen: PASS — both classes present")

# 3. Check combine_datasets deduplication logic with a small synthetic case
print("\n--- combine_datasets deduplication test ---")
from src.ingestion.dataset_loader import combine_datasets

df_a = pd.DataFrame({
    "label":     [1, 0, 1],
    "raw_email": ["Phishing email ABC def ghi", "Legitimate email 123", "Another phish xyz"],
    "source":    ["JinqiangDing/seven-phishing-email-datasets"] * 3,
})
df_b = pd.DataFrame({
    "label":     [1, 0],
    "raw_email": ["Phishing email ABC def ghi",  "Legitimate email 123"],  # duplicates of df_a
    "source":    ["zefang-liu/phishing-email-dataset"] * 2,
})
combined = combine_datasets(df_a, df_b)
# After dedup, should have 3 unique emails (JinqiangDing takes priority)
assert len(combined) == 3, f"FAIL: expected 3 rows after dedup, got {len(combined)}"
# The duplicates should come from JinqiangDing (priority=1) not zefang-liu (priority=5)
dup_sources = combined[combined["raw_email"].str.startswith("Phishing email ABC")]["source"].values
assert all(s == "JinqiangDing/seven-phishing-email-datasets" for s in dup_sources), \
    f"FAIL: priority dedup kept wrong source: {dup_sources}"
print(f"combine_datasets: PASS — 3 unique rows, JinqiangDing priority respected")

print("\n=== All 3 sanity checks PASSED ===")
