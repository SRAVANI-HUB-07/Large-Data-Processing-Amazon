#!/usr/bin/env python3
"""
Test script to verify data loading works
"""

import os
import sys
from pathlib import Path

# Set Python paths for Spark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.insert(0, str(Path(__file__).parent))

from config.spark_config import create_spark_session, stop_spark_session
from src.data_processing.data_loader import DataLoader

def test_data_loading():
    """Test basic data loading functionality"""
    print("Testing Amazon Data Loader...")
    
    spark = None
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Create data loader
        data_loader = DataLoader(spark)
        
        # Test loading just the raw data first
        print("Loading raw data...")
        data_path = data_loader.download_dataset()
        print(f"Data path: {data_path}")
        
        # Read raw data to test schema
        raw_df = spark.read \
            .option("multiLine", "true") \
            .option("mode", "DROPMALFORMED") \
            .option("allowSingleQuotes", "true") \
            .json(data_path)
        
        print(f"Raw data count: {raw_df.count():,}")
        print("Raw data schema:")
        raw_df.printSchema()
        
        # Show available columns
        print("Available columns:", raw_df.columns)
        
        # Show sample data
        print("Sample data:")
        raw_df.select("asin", "title", "categories", "salesRank").limit(5).show(truncate=50)
        
        print("✅ Data loading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            stop_spark_session(spark)

if __name__ == "__main__":
    test_data_loading()