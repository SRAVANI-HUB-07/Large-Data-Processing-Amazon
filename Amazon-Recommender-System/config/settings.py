import os
import urllib.request
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Dataset configuration
DATASET_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz"
DATASET_FILENAME = "amazon_books.json.gz"
COMPRESSED_FILENAME = "amazon_books.json.gz"

# Alternative datasets
ALTERNATIVE_DATASET_URLS = [
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz",
]

# Spark configuration
SPARK_APP_NAME = "AmazonRecommenderSystem"
SPARK_MASTER = "local[2]"
SPARK_EXECUTOR_MEMORY = "2g"
SPARK_DRIVER_MEMORY = "2g"

# Search configuration
MAX_SEARCH_RESULTS = 100
DEFAULT_TOP_N = 10

# Recommendation configuration
MIN_RATING_THRESHOLD = 4.0
MIN_REVIEWS_THRESHOLD = 10
SIMILARITY_THRESHOLD = 0.7

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def ensure_dataset_downloaded():
    """
    Ensure the dataset is downloaded and available
    Returns: Path to the dataset file
    """
    dataset_path = RAW_DATA_DIR / DATASET_FILENAME
    
    # Check if dataset exists and is valid
    if dataset_path.exists():
        size = dataset_path.stat().st_size
        if size > 1000000:  # At least 1MB
            print(f"✓ Dataset exists: {size:,} bytes")
            return dataset_path
        else:
            print(f"✗ Dataset too small ({size} bytes), re-downloading...")
            dataset_path.unlink()
    
    # Download the dataset
    print("📥 Downloading Amazon Books dataset...")
    print("This may take 5-10 minutes depending on your connection...")
    
    try:
        def progress_callback(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent}% ({mb_downloaded:.1f} / {mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(DATASET_URL, dataset_path, progress_callback)
        print("\n✓ Download completed!")
        
        # Verify download
        if dataset_path.exists():
            size = dataset_path.stat().st_size
            print(f"✓ Dataset ready: {size:,} bytes")
            return dataset_path
        else:
            raise Exception("Download failed - file not created")
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        raise