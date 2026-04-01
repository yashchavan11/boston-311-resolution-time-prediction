
"""
Step 1: Data Collection
=======================
Downloads 10 years (2015-2024) of Boston 311 service request data from the
City of Boston Open Data Portal (data.boston.gov).

Each year's data is a CSV with ~200K-280K records and 31 columns including
case details, dates, department, location, and status fields.

Output: data/raw/311_requests_{year}.csv (one file per year)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_seed, ensure_dirs
from src.data_loader import download_all_years, load_all_years

set_seed(42)
ensure_dirs()

# Download all years (skips files that already exist)
print("=" * 70)
print("  STEP 1: DATA COLLECTION")
print("=" * 70)
print("\nDownloading Boston 311 data from data.boston.gov...")
download_all_years(years=list(range(2015, 2025)))

# Verify downloads
print("\nVerifying loaded data...")
df = load_all_years(list(range(2015, 2025)))
print(f"\nTotal records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['open_dt'].min() if 'open_dt' in df.columns else 'N/A'} to "
      f"{df['open_dt'].max() if 'open_dt' in df.columns else 'N/A'}")
print("\nStep 1 complete. Raw data saved to data/raw/")
