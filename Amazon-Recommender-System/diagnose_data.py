#!/usr/bin/env python3
"""
Diagnose data loading issues
"""

import os
import sys
from pathlib import Path

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.insert(0, str(Path(__file__).parent))

from config.spark_config import create_spark_session, stop_spark_session

def diagnose_data():
    spark = None
    try:
        spark = create_spark_session()
        
        data_path = "C:/Users/johnii/Amazon-Recommender-System/data/raw/amazon_books.json"
        
        print("=== Data Diagnosis ===")
        
        # Try different reading approaches
        print("\n1. Basic read:")
        df1 = spark.read.option("mode", "DROPMALFORMED").json(data_path)
        print(f"   Records: {df1.count():,}")
        print(f"   Columns: {df1.columns}")
        
        print("\n2. With allowSingleQuotes:")
        df2 = spark.read \
            .option("mode", "DROPMALFORMED") \
            .option("allowSingleQuotes", "true") \
            .json(data_path)
        print(f"   Records: {df2.count():,}")
        print(f"   Columns: {df2.columns}")
        
        print("\n3. Sample data:")
        df2.limit(3).show(truncate=50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            stop_spark_session(spark)

if __name__ == "__main__":
    diagnose_data()