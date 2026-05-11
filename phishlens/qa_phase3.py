"""Phase 3 QA: Test absent HF dataset loaders in isolation and check eval exit code."""
import sys
sys.path.insert(0, ".")
from src.utils.logger import configure_logger
configure_logger(level="INFO")

from src.ingestion.dataset_loader import load_hf_zefang, load_hf_puyang_seven

print("=" * 60)
print("TEST 1: zefang-liu/phishing-email-dataset")
print("=" * 60)
df_z = load_hf_zefang()
print(f"Rows returned: {len(df_z):,}")
if len(df_z) > 0:
    print(f"  Phishing: {(df_z['label']==1).sum():,}")
    print(f"  Legit   : {(df_z['label']==0).sum():,}")
    print(f"  Columns : {list(df_z.columns)}")
    print(f"  Sample  : {df_z['raw_email'].iloc[0][:80]!r}")

print()
print("=" * 60)
print("TEST 2: puyang2025/seven-phishing-email-datasets")
print("=" * 60)
df_p = load_hf_puyang_seven()
print(f"Rows returned: {len(df_p):,}")
if len(df_p) > 0:
    print(f"  Phishing: {(df_p['label']==1).sum():,}")
    print(f"  Legit   : {(df_p['label']==0).sum():,}")
    print(f"  First 3 emails (50 chars):")
    for i, row in df_p.head(3).iterrows():
        print(f"    [{i}] {str(row['raw_email'])[:50]!r}")
