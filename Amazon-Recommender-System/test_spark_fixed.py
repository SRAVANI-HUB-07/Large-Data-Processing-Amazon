#!/usr/bin/env python3
"""Fixed Spark test for Windows"""
import os
import sys

# Set Python paths explicitly
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

try:
    from pyspark.sql import SparkSession
    
    # Create minimal Spark session
    spark = SparkSession.builder \
        .appName("FixedTest") \
        .master("local[1]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .getOrCreate()
    
    print("✓ Spark session created successfully!")
    
    # Simple test without complex operations
    data = [("Test", 1), ("Data", 2)]
    df = spark.createDataFrame(data, ["Name", "Value"])
    print(f"✓ DataFrame created with {df.count()} rows")
    df.show()
    
    spark.stop()
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()