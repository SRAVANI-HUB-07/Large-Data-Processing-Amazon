import logging
import time
from functools import wraps
from typing import Any, Callable
from pyspark.sql import DataFrame, functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('amazon_recommender.log'),
            logging.StreamHandler()
        ]
    )

def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def safe_division(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0

def sample_dataframe(df: DataFrame, fraction: float = 0.1, seed: int = 42) -> DataFrame:
    return df.sample(fraction=fraction, seed=seed)

def save_dataframe_to_csv(df: DataFrame, path: str, single_file: bool = True):
    if single_file:
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
    else:
        df.write.mode("overwrite").option("header", "true").csv(path)

def create_spark_dataframe_summary(df: DataFrame) -> dict:
    summary = {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": df.columns,
        "schema": str(df.schema),
        "null_counts": {col: df.filter(F.col(col).isNull()).count() for col in df.columns}
    }
    
    numeric_cols = [field.name for field in df.schema.fields 
                   if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
    
    if numeric_cols:
        stats = df.select([F.mean(col).alias(f"mean_{col}") for col in numeric_cols] +
                         [F.stddev(col).alias(f"std_{col}") for col in numeric_cols]).collect()[0]
        summary["statistics"] = stats.asDict()
    
    return summary

def plot_recommendation_metrics(metrics: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Recommendation System Evaluation Metrics')
    
    if 'rmse' in metrics and 'mae' in metrics:
        error_metrics = ['RMSE', 'MAE']
        error_values = [metrics['rmse'], metrics['mae']]
        axes[0, 0].bar(error_metrics, error_values, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Error Metrics')
        axes[0, 0].set_ylabel('Value')
    
    if 'precision' in metrics and 'recall' in metrics:
        pr_metrics = ['Precision', 'Recall']
        pr_values = [metrics['precision'], metrics['recall']]
        axes[0, 1].bar(pr_metrics, pr_values, color=['lightgreen', 'gold'])
        axes[0, 1].set_title('Precision and Recall')
        axes[0, 1].set_ylabel('Value')
    
    if 'category_distribution' in metrics:
        categories = list(metrics['category_distribution'].keys())[:10]
        counts = list(metrics['category_distribution'].values())[:10]
        axes[1, 0].pie(counts, labels=categories, autopct='%1.1f%%')
        axes[1, 0].set_title('Top 10 Categories Distribution')
    
    if 'rating_distribution' in metrics:
        ratings = list(metrics['rating_distribution'].keys())
        counts = list(metrics['rating_distribution'].values())
        axes[1, 1].bar(ratings, counts, color='purple', alpha=0.7)
        axes[1, 1].set_title('Rating Distribution')
        axes[1, 1].set_xlabel('Rating')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def validate_search_parameters(category: str = None, min_rating: float = None, 
                             min_reviews: int = None, n: int = None) -> bool:
    if category and not isinstance(category, str):
        raise ValueError("Category must be a string")
    
    if min_rating and (min_rating < 0 or min_rating > 5):
        raise ValueError("Minimum rating must be between 0 and 5")
    
    if min_reviews and min_reviews < 0:
        raise ValueError("Minimum reviews cannot be negative")
    
    if n and n <= 0:
        raise ValueError("Number of results must be positive")
    
    return True

def format_recommendations_for_display(recommendations_df: DataFrame, 
                                     products_df: DataFrame) -> DataFrame:
    return recommendations_df.join(products_df, "asin") \
        .select("asin", "title", "group", "avg_rating", "reviews_total", "salesrank") \
        .orderBy(F.desc("avg_rating"), F.asc("salesrank"))