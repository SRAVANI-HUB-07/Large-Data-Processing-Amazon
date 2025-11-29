#!/usr/bin/env python3
"""Test Spark installation"""
import sys

try:
    from pyspark.sql import SparkSession
    print("✓ PySpark imported successfully")
    
    # Test Spark session
    spark = SparkSession.builder \
        .appName("Test") \
        .master("local[*]") \
        .getOrCreate()
    
    # Test basic functionality
    data = [("John", 25), ("Jane", 30)]
    df = spark.createDataFrame(data, ["Name", "Age"])
    print(f"✓ Spark session created successfully")
    print(f"✓ DataFrame count: {df.count()}")
    df.show()
    
    spark.stop()
    print("✓ All Spark tests passed!")
    
except Exception as e:
    print(f"✗ Spark test failed: {e}")
    sys.exit(1)