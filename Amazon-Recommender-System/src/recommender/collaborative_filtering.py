from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """
    Collaborative Filtering based on user-item interaction patterns
    Implements 'customers who bought this also bought' feature
    """
    
    def __init__(self, spark, products_df: DataFrame, reviews_df: DataFrame):
        self.spark = spark
        self.products_df = products_df
        self.reviews_df = reviews_df
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.trained = False
    
    def train_model(self):
        """Train collaborative filtering model"""
        logger.info("Training collaborative filtering model...")
        
        try:
            # Create user-item interaction matrix
            self._create_user_item_matrix()
            
            # Calculate item-item similarities
            self._calculate_item_similarities()
            
            self.trained = True
            logger.info("Collaborative filtering model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            raise
    
    def _create_user_item_matrix(self):
        """Create user-item interaction matrix from reviews"""
        logger.info("Creating user-item interaction matrix...")
        
        # Create binary interactions (1 if user reviewed product, 0 otherwise)
        self.user_item_matrix = self.reviews_df \
            .select("customer_id", "product_id", "rating") \
            .withColumn("interaction", F.lit(1)) \
            .groupBy("customer_id", "product_id") \
            .agg(F.first("interaction").alias("interaction"))
        
        logger.info(f"User-item matrix created: {self.user_item_matrix.count()} interactions")
    
    def _calculate_item_similarities(self):
        """Calculate item-item similarities using co-occurrence patterns"""
        logger.info("Calculating item-item similarities...")
        
        # Create product pairs that were bought by same users
        user_products = self.user_item_matrix.alias("up1")
        product_pairs = user_products \
            .join(
                self.user_item_matrix.alias("up2"),
                (F.col("up1.customer_id") == F.col("up2.customer_id")) &
                (F.col("up1.product_id") != F.col("up2.product_id"))
            ) \
            .select(
                F.col("up1.product_id").alias("product1"),
                F.col("up2.product_id").alias("product2"),
                F.col("up1.interaction").alias("interaction1"),
                F.col("up2.interaction").alias("interaction2")
            )
        
        # Calculate similarity scores (Jaccard similarity)
        product_pair_scores = product_pairs \
            .groupBy("product1", "product2") \
            .agg(F.count("*").alias("co_occurrence_count"))
        
        # Calculate individual product frequencies
        product_freq = self.user_item_matrix \
            .groupBy("product_id") \
            .agg(F.count("*").alias("product_frequency"))
        
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        similarity_df = product_pair_scores \
            .join(
                product_freq.alias("f1"),
                F.col("product1") == F.col("f1.product_id")
            ) \
            .join(
                product_freq.alias("f2"),
                F.col("product2") == F.col("f2.product_id")
            ) \
            .withColumn(
                "similarity_score",
                F.col("co_occurrence_count") / 
                (F.col("f1.product_frequency") + F.col("f2.product_frequency") - F.col("co_occurrence_count"))
            ) \
            .select("product1", "product2", "similarity_score")
        
        self.item_similarity_matrix = similarity_df
        logger.info(f"Item similarity matrix created: {self.item_similarity_matrix.count()} pairs")
    
    def get_also_bought_products(self, product_id: str, n: int = 10) -> DataFrame:
        """
        Get 'customers who bought this also bought' recommendations
        Based on item-item collaborative filtering
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Finding products also bought with {product_id}")
        
        # Get similar products based on co-occurrence
        similar_products = self.item_similarity_matrix \
            .filter(F.col("product1") == product_id) \
            .orderBy(F.desc("similarity_score")) \
            .limit(n)
        
        # Join with product details
        result = similar_products \
            .join(
                self.products_df.alias("p"),
                F.col("product2") == F.col("p.product_id")
            ) \
            .select(
                "p.product_id",
                "p.title",
                "p.category",
                "p.average_rating",
                "p.review_count",
                "p.price",
                "similarity_score"
            ) \
            .orderBy(F.desc("similarity_score"))
        
        return result
    
    def recommend_for_user(self, user_id: str, n: int = 10) -> DataFrame:
        """
        Get recommendations for a user based on their purchase history
        using user-based collaborative filtering
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Generating collaborative recommendations for user {user_id}")
        
        # Get user's purchased products
        user_products = self.user_item_matrix \
            .filter(F.col("customer_id") == user_id) \
            .select("product_id") \
            .collect()
        
        if not user_products:
            logger.warning(f"No purchase history found for user {user_id}")
            return self.get_popular_products(n)
        
        user_product_ids = [row.product_id for row in user_products]
        
        # Find similar users based on product overlap
        similar_users = self.user_item_matrix \
            .filter(
                (F.col("customer_id") != user_id) &
                (F.col("product_id").isin(user_product_ids))
            ) \
            .groupBy("customer_id") \
            .agg(F.count("*").alias("common_products")) \
            .orderBy(F.desc("common_products")) \
            .limit(100)
        
        # Get products from similar users that the target user hasn't bought
        recommendations = similar_users \
            .join(
                self.user_item_matrix.alias("ui"),
                "customer_id"
            ) \
            .filter(~F.col("ui.product_id").isin(user_product_ids)) \
            .groupBy("ui.product_id") \
            .agg(
                F.count("*").alias("recommendation_score"),
                F.avg("common_products").alias("user_similarity")
            ) \
            .orderBy(F.desc("recommendation_score")) \
            .limit(n)
        
        # Join with product details
        result = recommendations \
            .join(
                self.products_df.alias("p"),
                F.col("ui.product_id") == F.col("p.product_id")
            ) \
            .select(
                "p.product_id",
                "p.title",
                "p.category",
                "p.average_rating",
                "p.review_count",
                "p.price",
                "recommendation_score",
                "user_similarity"
            ) \
            .orderBy(F.desc("recommendation_score"))
        
        return result
    
    def get_similar_products(self, product_id: str, n: int = 10) -> DataFrame:
        """
        Get similar products based on collaborative filtering
        Alternative implementation using direct similarity
        """
        return self.get_also_bought_products(product_id, n)
    
    def get_popular_products(self, n: int = 10) -> DataFrame:
        """
        Get popular products based on number of interactions
        """
        popular_products = self.user_item_matrix \
            .groupBy("product_id") \
            .agg(F.count("*").alias("interaction_count")) \
            .orderBy(F.desc("interaction_count")) \
            .limit(n)
        
        result = popular_products \
            .join(
                self.products_df.alias("p"),
                "product_id"
            ) \
            .select(
                "p.product_id",
                "p.title",
                "p.category",
                "p.average_rating",
                "p.review_count",
                "p.price",
                "interaction_count"
            ) \
            .orderBy(F.desc("interaction_count"))
        
        return result
    
    def get_user_similarity(self, user1: str, user2: str) -> float:
        """
        Calculate similarity between two users based on their product interactions
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get products for each user
        user1_products = self.user_item_matrix \
            .filter(F.col("customer_id") == user1) \
            .select("product_id") \
            .rdd.flatMap(lambda x: x).collect()
        
        user2_products = self.user_item_matrix \
            .filter(F.col("customer_id") == user2) \
            .select("product_id") \
            .rdd.flatMap(lambda x: x).collect()
        
        if not user1_products or not user2_products:
            return 0.0
        
        # Calculate Jaccard similarity
        set1 = set(user1_products)
        set2 = set(user2_products)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_model(self, test_ratio: float = 0.2) -> Dict[str, float]:
        """
        Evaluate collaborative filtering model using train-test split
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info("Evaluating collaborative filtering model...")
        
        # Split data into train and test
        train_df, test_df = self.user_item_matrix.randomSplit([1 - test_ratio, test_ratio])
        
        # For simplicity, we'll use precision@k as our metric
        # In a real scenario, we would implement more comprehensive evaluation
        
        test_users = test_df.select("customer_id").distinct().limit(100).collect()
        
        precision_scores = []
        
        for test_user in test_users[:10]:  # Sample for performance
            user_id = test_user.customer_id
            
            try:
                # Get actual products from test set
                actual_products = test_df \
                    .filter(F.col("customer_id") == user_id) \
                    .select("product_id") \
                    .rdd.flatMap(lambda x: x).collect()
                
                if not actual_products:
                    continue
                
                # Get recommendations
                recommendations = self.recommend_for_user(user_id, 10)
                recommended_products = recommendations \
                    .select("product_id") \
                    .rdd.flatMap(lambda x: x).collect()
                
                if not recommended_products:
                    continue
                
                # Calculate precision
                hits = len(set(actual_products).intersection(set(recommended_products)))
                precision = hits / len(recommended_products)
                precision_scores.append(precision)
                
            except Exception as e:
                logger.debug(f"Error evaluating user {user_id}: {e}")
                continue
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        
        metrics = {
            "precision_at_10": avg_precision,
            "users_evaluated": len(precision_scores),
            "test_users_total": len(test_users)
        }
        
        logger.info(f"Model evaluation completed: Precision@10 = {avg_precision:.4f}")
        return metrics