import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from config.settings import SPARK_APP_NAME, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY

def create_spark_session():
    """Create and configure Spark session with Windows compatibility"""
    
    # Set Python environment explicitly
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    # Optimized configuration for Windows
    conf = SparkConf().setAppName(SPARK_APP_NAME) \
                      .setMaster(SPARK_MASTER) \
                      .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
                      .set("spark.driver.memory", SPARK_DRIVER_MEMORY) \
                      .set("spark.sql.adaptive.enabled", "false") \
                      .set("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                      .set("spark.dynamicAllocation.enabled", "false") \
                      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                      .set("spark.sql.legacy.timeParserPolicy", "LEGACY") \
                      .set("spark.python.worker.reuse", "false") \
                      .set("spark.sql.autoBroadcastJoinThreshold", "-1") \
                      .set("spark.sql.join.preferSortMergeJoin", "true") \
                      .set("spark.default.parallelism", "1") \
                      .set("spark.sql.shuffle.partitions", "1") \
                      .set("spark.task.cpus", "1") \
                      .set("spark.executor.cores", "1") \
                      .set("spark.network.timeout", "800s") \
                      .set("spark.executor.heartbeatInterval", "120s") \
                      .set("spark.sql.parquet.compression.codec", "uncompressed") \
                      .set("spark.memory.fraction", "0.6") \
                      .set("spark.memory.storageFraction", "0.3")
    
    spark = SparkSession.builder \
                       .config(conf=conf) \
                       .getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark

def stop_spark_session(spark):
    """Stop Spark session gracefully"""
    if spark:
        try:
            spark.stop()
        except Exception:
            pass