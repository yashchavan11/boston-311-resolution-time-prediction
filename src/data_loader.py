"""Data loading and download utilities for Boston 311 dataset."""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

from src.utils import DATA_RAW

# -- Boston 311 Resource IDs (data.boston.gov CKAN) --------------------------
# These resource IDs map to per-year CSVs on the Boston Open Data portal.
# Updated March 2026. If a download fails, check data.boston.gov for new IDs.
RESOURCE_IDS = {
    2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
    2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
    2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
    2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
    2020: "6ff6a6fd-3141-4440-a880-6f60a37fe789",
    2019: "ea2e4696-4a2d-429c-9807-d02eb92e0222",
    2018: "2be28d90-3a90-4af1-a3f6-f28c1e25880a",
    2017: "30022137-709d-465e-baae-ca155b51927d",
    2016: "b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a",
    2015: "c9509ab4-6f6d-4b97-979a-0cf2a10c922b",
}

CKAN_API_BASE = "https://data.boston.gov/api/3/action/resource_show?id="


def get_download_url(resource_id: str) -> str:
    """Get the actual download URL for a resource via CKAN API."""
    try:
        resp = requests.get(f"{CKAN_API_BASE}{resource_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["result"]["url"]
    except Exception as e:
        # Fallback: construct URL directly
        return f"https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/{resource_id}/download/311_service_requests_{resource_id[:8]}.csv"


def download_year(year: int, data_dir: Optional[Path] = None) -> Path:
    """
    Download a single year of Boston 311 data from data.boston.gov.

    Parameters
    ----------
    year : int
        Year to download (2015-2024).
    data_dir : Path, optional
        Directory to save the file. Defaults to data/raw/.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    if data_dir is None:
        data_dir = DATA_RAW
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / f"311_requests_{year}.csv"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Already downloaded: {out_path.name} ({size_mb:.1f} MB)")
        return out_path

    if year not in RESOURCE_IDS:
        raise ValueError(f"No resource ID for year {year}. Available: {sorted(RESOURCE_IDS.keys())}")

    resource_id = RESOURCE_IDS[year]
    download_url = get_download_url(resource_id)

    print(f"  Downloading {year} from {download_url[:80]}...")
    try:
        r = requests.get(download_url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"{year}") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path.name} ({size_mb:.1f} MB)")
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Failed to download year {year}: {e}")

    return out_path


def download_all_years(years: Optional[List[int]] = None, data_dir: Optional[Path] = None) -> List[Path]:
    """Download multiple years of 311 data."""
    if years is None:
        years = sorted(RESOURCE_IDS.keys())

    paths = []
    for year in years:
        try:
            p = download_year(year, data_dir)
            paths.append(p)
        except Exception as e:
            print(f"  WARNING: Could not download {year}: {e}")
    return paths


def load_year(year: int, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a single year CSV into a DataFrame."""
    if data_dir is None:
        data_dir = DATA_RAW
    path = Path(data_dir) / f"311_requests_{year}.csv"

    if not path.exists():
        print(f"  File not found for {year}, attempting download...")
        download_year(year, data_dir)

    df = pd.read_csv(path, low_memory=False)
    df["source_year"] = year
    return df


def load_all_years(years: List[int], data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and concatenate multiple years of Boston 311 data.

    Parameters
    ----------
    years : list of int
        Years to load.
    data_dir : Path, optional
        Directory containing the CSV files. Defaults to data/raw/.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a 'source_year' column.
    """
    frames = []
    for year in sorted(years):
        print(f"Loading {year}...")
        try:
            df = load_year(year, data_dir)
            print(f"  {len(df):,} rows, {len(df.columns)} columns")
            frames.append(df)
        except Exception as e:
            print(f"  ERROR loading {year}: {e}")

    if not frames:
        raise RuntimeError("No data loaded! Check your data directory and downloads.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(combined):,} rows across {len(frames)} years")
    return combined


def load_processed(split: str = "train") -> pd.DataFrame:
    """Load a processed parquet file (train/val/test)."""
    from src.utils import DATA_PROCESSED
    path = DATA_PROCESSED / f"{split}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run notebook 03_feature_engineering.ipynb first."
        )
    return pd.read_parquet(path)
