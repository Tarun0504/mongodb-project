"""
CineScope Dataset Download Script
Downloads the MovieLens Latest Small dataset.
"""
import os
import sys
import zipfile
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_URL, DATA_DIR


def download_dataset():
    """Download and extract the MovieLens dataset."""
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    extract_dir = os.path.join(DATA_DIR, "ml-latest-small")

    # Check if already extracted
    required_files = ["movies.csv", "ratings.csv"]
    all_present = all(
        os.path.exists(os.path.join(extract_dir, f)) for f in required_files
    )

    if all_present:
        print("✓ Dataset already downloaded and extracted")
        return extract_dir

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download if ZIP not present
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens dataset from:\n  {DATASET_URL}")
        print("This may take a moment...")
        try:
            response = requests.get(DATASET_URL, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
            print(f"\n✓ Downloaded: {zip_path} ({downloaded:,} bytes)")
        except requests.RequestException as e:
            print(f"\n✗ Download failed: {e}")
            print(f"\nManual download instructions:")
            print(f"  1. Visit: https://grouplens.org/datasets/movielens/")
            print(f"  2. Download 'ml-latest-small.zip'")
            print(f"  3. Place it in: {DATA_DIR}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            sys.exit(1)
    else:
        print(f"✓ ZIP archive found: {zip_path}")

    # Extract
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"✓ Extracted to: {extract_dir}")
    except zipfile.BadZipFile:
        print(f"✗ Corrupt ZIP file. Delete {zip_path} and re-run.")
        sys.exit(1)

    # Verify
    for f in required_files:
        fpath = os.path.join(extract_dir, f)
        if not os.path.exists(fpath):
            print(f"✗ Missing required file: {f}")
            sys.exit(1)
    
    print("✓ All required files verified")
    return extract_dir


if __name__ == "__main__":
    download_dataset()
