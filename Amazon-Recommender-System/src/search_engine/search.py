from pyspark.sql import DataFrame, functions as F
from typing import Dict, Any
from .queries import SearchQueries

class SearchEngine:
    def __init__(self, products_df: DataFrame, reviews_df: DataFrame):
        self.products_df = products_df
        self.reviews_df = reviews_df
        self.queries = SearchQueries()
    
    def search(self, query_type: str, **kwargs) -> DataFrame:
        """
        Execute search queries with enhanced functionality
        """
        if query_type == "best_sellers":
            return self.queries.best_sellers(
                self.products_df, 
                kwargs['category'], 
                kwargs.get('n', 10)
            )
        
        elif query_type == "review_count":
            return self.queries.products_by_review_count(
                self.products_df,
                kwargs['operator'],
                kwargs['count_threshold']
            )
        
        elif query_type == "rating":
            return self.queries.products_by_rating(
                self.products_df,
                kwargs['operator'],
                kwargs['rating_threshold'],
                kwargs.get('n', 10)
            )
        
        elif query_type == "complex":
            return self.queries.complex_queries(
                self.products_df,
                kwargs.get('category'),
                kwargs.get('min_rating'),
                kwargs.get('min_reviews'),
                kwargs.get('max_salesrank')
            )
        
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
    
    def get_product_info(self, product_id: str) -> DataFrame:
        """
        Get detailed information about a specific product
        Returns all available product information including review count
        """
        product_info = self.products_df.filter(F.col("product_id") == product_id)
        
        # Calculate review count for the product
        review_count = self.reviews_df.filter(F.col("product_id") == product_id).count()
        
        # Add review count column
        return product_info.withColumn("review_count", F.lit(review_count))
    
    def get_user_review_history(self, user_id: str) -> DataFrame:
        """
        Get all reviews by a specific user with product details including category
        """
        user_reviews = self.reviews_df.filter(F.col("customer_id") == user_id)
        
        if user_reviews.count() == 0:
            # Return empty DataFrame with proper schema including category
            return self.reviews_df.filter(F.col("customer_id") == "none").join(
                self.products_df.select("product_id", "title", "category"), 
                "product_id", 
                "left"
            )
        
        # Join with products to get product details including category
        return user_reviews.join(
            self.products_df.select("product_id", "title", "category"), 
            "product_id", 
            "inner"
        ).select(
            "product_id", 
            "title", 
            "category",
            "rating", 
            "review_date", 
            "helpful_votes",
            "total_votes"
        ).orderBy(F.desc("review_date"))
    
    def get_category_stats(self) -> DataFrame:
        """
        Get comprehensive statistics for each category
        """
        return self.products_df.groupBy("category").agg(
            F.count("product_id").alias("product_count"),
            F.avg("average_rating").alias("avg_category_rating"),
            F.avg("review_count").alias("avg_reviews_per_product"),
            F.min("sales_rank").alias("best_salesrank"),
            F.avg("price").alias("avg_price"),
            F.max("average_rating").alias("max_rating"),
            F.min("average_rating").alias("min_rating")
        ).orderBy(F.desc("product_count"))
    
    def get_top_products(self, category: str = None, n: int = 10) -> DataFrame:
        """
        Get top products by rating and popularity
        """
        base_df = self.products_df
        
        if category:
            base_df = base_df.filter(F.col("category") == category)
        
        return base_df.filter(
            (F.col("average_rating") >= 4.0) &
            (F.col("review_count") > 0)
        ).orderBy(
            F.desc("average_rating"), 
            F.asc("sales_rank")
        ).limit(n)
    
    def find_similar_products(self, product_id: str, n: int = 10) -> DataFrame:
        """
        Find products similar to a given product
        Based on category and rating similarity
        """
        # Get the target product
        target_product = self.products_df.filter(
            F.col("product_id") == product_id
        )
        
        if target_product.count() == 0:
            return self.products_df.limit(0)
        
        target_row = target_product.first()
        target_category = target_row["category"]
        target_rating = target_row["average_rating"]
        
        # Find similar products in same category with similar ratings
        similar_products = self.products_df.filter(
            (F.col("category") == target_category) &
            (F.col("product_id") != product_id) &
            (F.col("average_rating") >= target_rating - 0.5) &
            (F.col("average_rating") <= target_rating + 0.5)
        ).orderBy(
            F.desc("average_rating"),
            F.asc("sales_rank")
        ).limit(n)
        
        return similar_products
    
    def get_copurchasers_count(self, user_id: str, product_id: str) -> int:
        """
        Get the number of customers purchasing the same product given user_id and product_id
        If dataset has any number give that else give zero
        """
        try:
            # Count distinct customers who purchased the same product, excluding the given user
            count = self.reviews_df.filter(
                (F.col("product_id") == product_id) & 
                (F.col("customer_id") != user_id)
            ).select("customer_id").distinct().count()
            
            return count
        except Exception:
            return 0
    
    def get_popular_products(self, n: int = 10) -> DataFrame:
        """
        Get popular products based on sales rank and rating
        """
        return self.products_df.filter(
            F.col("sales_rank") < 1000000
        ).orderBy(
            F.asc("sales_rank"),
            F.desc("average_rating")
        ).limit(n)
    
    def search_by_title(self, search_term: str, n: int = 10) -> DataFrame:
        """
        Search products by title using partial matching
        """
        return self.products_df.filter(
            F.lower(F.col("title")).contains(search_term.lower())
        ).orderBy(
            F.asc("sales_rank")
        ).limit(n)
    
    def get_products_by_price_range(self, min_price: float = 0, max_price: float = 1000, n: int = 10) -> DataFrame:
        """
        Get products within a specific price range
        """
        return self.products_df.filter(
            (F.col("price") >= min_price) &
            (F.col("price") <= max_price)
        ).orderBy(
            F.asc("sales_rank")
        ).limit(n)
    
    def get_products_with_high_engagement(self, min_reviews: int = 10, n: int = 10) -> DataFrame:
        """
        Get products with high engagement (many reviews)
        """
        return self.products_df.filter(
            F.col("review_count") >= min_reviews
        ).orderBy(
            F.desc("review_count"),
            F.desc("average_rating")
        ).limit(n)