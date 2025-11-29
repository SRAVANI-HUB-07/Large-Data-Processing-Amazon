from pyspark.sql import DataFrame, functions as F
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, spark, products_df: DataFrame, reviews_df: DataFrame):
        self.spark = spark
        self.products_df = products_df
        self.reviews_df = reviews_df
        
    def popularity_based_recommendations(self, category: str = None, n: int = 10) -> DataFrame:
        """Recommend popular products based on sales rank and ratings"""
        base_df = self.products_df
        
        if category:
            base_df = base_df.filter(F.col("category") == category)
        
        # Filter out products with no reviews and poor ratings
        return base_df.filter(
            (F.col("average_rating") >= 3.5) |
            (F.col("sales_rank") <= 100000)
        ).orderBy(
            F.asc("sales_rank"), 
            F.desc("average_rating")
        ).limit(n)
    
    def content_based_recommendations(self, user_id: str, n: int = 10) -> DataFrame:
        """Recommend products based on user's preferred categories"""
        # Get user's review history
        user_reviews = self.reviews_df.filter(F.col("customer_id") == user_id)
        
        if user_reviews.count() == 0:
            logger.warning(f"No reviews found for user {user_id}")
            return self.popularity_based_recommendations(n=n)
        
        # Get user's preferred categories from their reviews
        user_categories = user_reviews.join(self.products_df, "product_id") \
            .groupBy("category") \
            .agg(
                F.count("rating").alias("review_count"),
                F.avg("rating").alias("avg_user_rating")
            ) \
            .filter(F.col("review_count") >= 1) \
            .orderBy(F.desc("review_count"), F.desc("avg_user_rating"))
        
        category_list = [row.category for row in user_categories.collect()]
        
        if not category_list:
            return self.popularity_based_recommendations(n=n)
        
        # Recommend highly-rated products in user's preferred categories
        return self.products_df.filter(
            F.col("category").isin(category_list) &
            (F.col("average_rating") >= 4.0)
        ).orderBy(
            F.desc("average_rating"), 
            F.asc("sales_rank")
        ).limit(n)
    
    def hybrid_recommendations(self, user_id: str, n: int = 10) -> DataFrame:
        """Combine content-based and popularity-based recommendations"""
        content_recs = self.content_based_recommendations(user_id, n)
        
        if content_recs.count() >= n:
            return content_recs
        else:
            # If not enough content-based recs, add popular ones
            content_count = content_recs.count()
            remaining = n - content_count
            
            if remaining > 0:
                popular_recs = self.popularity_based_recommendations(n=remaining)
                return content_recs.union(popular_recs)
            else:
                return content_recs
    
    def similar_products(self, product_id: str, n: int = 10) -> DataFrame:
        """Find products similar to the given product"""
        try:
            # Find product info
            product_info = self.products_df.filter(F.col("product_id") == product_id)
            
            if product_info.count() == 0:
                logger.warning(f"Product {product_id} not found")
                return self.popularity_based_recommendations(n=n)
            
            product_row = product_info.first()
            product_category = product_row["category"]
            
            # First, try to use similar products from the data
            similar_products_list = product_row["similar_products"]
            
            if similar_products_list and len(similar_products_list) > 0:
                # Use the pre-computed similar products
                similar_df = self.products_df.filter(
                    F.col("product_id").isin(similar_products_list)
                ).limit(n)
                
                if similar_df.count() > 0:
                    return similar_df
            
            # Fallback: find similar products in same category with good ratings
            return self.products_df.filter(
                (F.col("category") == product_category) &
                (F.col("product_id") != product_id) &
                (F.col("average_rating") >= 3.5)
            ).orderBy(
                F.desc("average_rating"), 
                F.asc("sales_rank")
            ).limit(n)
            
        except Exception as e:
            logger.error(f"Error in similar_products: {e}")
            # Fallback to popular products
            return self.popularity_based_recommendations(n=n)

class RecommendationEvaluator:
    @staticmethod
    def precision_at_k(actual: List[str], predicted: List[str], k: int = 10) -> float:
        """Calculate precision@K for recommendations"""
        if not actual or not predicted:
            return 0.0
        
        actual_set = set(actual[:k])
        predicted_set = set(predicted[:k])
        
        if not predicted_set:
            return 0.0
        
        return len(actual_set.intersection(predicted_set)) / len(predicted_set)
    
    @staticmethod
    def recall_at_k(actual: List[str], predicted: List[str], k: int = 10) -> float:
        """Calculate recall@K for recommendations"""
        if not actual or not predicted:
            return 0.0
        
        actual_set = set(actual[:k])
        predicted_set = set(predicted[:k])
        
        if not actual_set:
            return 0.0
        
        return len(actual_set.intersection(predicted_set)) / len(actual_set)
    
    @staticmethod
    def calculate_diversity(recommendations: DataFrame) -> float:
        """Calculate category diversity of recommendations"""
        if recommendations.count() == 0:
            return 0.0
        
        category_counts = recommendations.groupBy("category").count()
        total_categories = category_counts.count()
        total_products = recommendations.count()
        
        return total_categories / total_products if total_products > 0 else 0.0
    
    @staticmethod
    def calculate_avg_rating(recommendations: DataFrame) -> float:
        """Calculate average rating of recommended products"""
        if recommendations.count() == 0:
            return 0.0
        
        avg_rating = recommendations.agg(F.avg("average_rating")).collect()[0][0]
        return avg_rating or 0.0