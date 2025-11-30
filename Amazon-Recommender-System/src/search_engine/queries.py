from pyspark.sql import DataFrame, functions as F
from typing import List, Union

class SearchQueries:
    @staticmethod
    def best_sellers(df: DataFrame, category: str, n: int = 10) -> DataFrame:
        """
        Get best-selling products in a category
        """
        return df.filter(F.col("category") == category) \
                .orderBy(F.asc("sales_rank")) \
                .limit(n)
    
    @staticmethod
    def products_by_review_count(df: DataFrame, operator: str, count_threshold: int) -> DataFrame:
        """
        Filter products by review count
        """
        if operator == '>':
            return df.filter(F.col("review_count") > count_threshold)
        elif operator == '>=':
            return df.filter(F.col("review_count") >= count_threshold)
        elif operator == '=':
            return df.filter(F.col("review_count") == count_threshold)
        elif operator == '<':
            return df.filter(F.col("review_count") < count_threshold)
        elif operator == '<=':
            return df.filter(F.col("review_count") <= count_threshold)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    @staticmethod
    def products_by_rating(df: DataFrame, operator: str, rating_threshold: float, n: int = 10) -> DataFrame:
        """
        Filter products by average rating
        """
        if operator == '>':
            filtered_df = df.filter(F.col("average_rating") > rating_threshold)
        elif operator == '>=':
            filtered_df = df.filter(F.col("average_rating") >= rating_threshold)
        elif operator == '=':
            filtered_df = df.filter(F.col("average_rating") == rating_threshold)
        elif operator == '<':
            filtered_df = df.filter(F.col("average_rating") < rating_threshold)
        elif operator == '<=':
            filtered_df = df.filter(F.col("average_rating") <= rating_threshold)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
        # Apply limit if specified
        if n:
            filtered_df = filtered_df.limit(n)
            
        return filtered_df
    
    @staticmethod
    def complex_queries(products_df: DataFrame, 
                       category: str = None,
                       min_rating: float = None,
                       min_reviews: int = None,
                       max_salesrank: int = None,
                       max_price: float = None) -> DataFrame:
        """
        Complex queries with multiple filters
        """
        filtered_df = products_df
        
        if category:
            filtered_df = filtered_df.filter(F.col("category") == category)
        
        if min_rating:
            filtered_df = filtered_df.filter(F.col("average_rating") >= min_rating)
        
        if min_reviews:
            filtered_df = filtered_df.filter(F.col("review_count") >= min_reviews)
        
        if max_salesrank:
            filtered_df = filtered_df.filter(F.col("sales_rank") <= max_salesrank)
        
        if max_price:
            filtered_df = filtered_df.filter(F.col("price") <= max_price)
        
        return filtered_df.orderBy(F.asc("sales_rank"))
    
    @staticmethod
    def title_search(df: DataFrame, search_term: str) -> DataFrame:
        """
        Search products by title
        """
        return df.filter(
            F.lower(F.col("title")).contains(search_term.lower())
        )
    
    @staticmethod
    def price_range_search(df: DataFrame, min_price: float, max_price: float) -> DataFrame:
        """
        Search products by price range
        """
        return df.filter(
            (F.col("price") >= min_price) & 
            (F.col("price") <= max_price)
        )