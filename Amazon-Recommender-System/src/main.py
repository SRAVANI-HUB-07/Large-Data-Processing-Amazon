#!/usr/bin/env python3
"""
Enhanced Amazon Recommender System with Collaborative Filtering
"""

import os
import sys
import logging
from pathlib import Path

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session, stop_spark_session
from src.data_processing.data_loader import DataLoader
from src.search_engine.search import SearchEngine
from src.recommender.algorithms import HybridRecommender
from src.recommender.collaborative_filtering import CollaborativeFiltering
from src.utils.helpers import setup_logging

class AmazonRecommenderApp:
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.spark = None
        self.data_loader = None
        self.products_df = None
        self.reviews_df = None
        self.search_engine = None
        self.recommender = None
        self.collab_filter = None
        
    def initialize_data(self):
        """Initialize data and components"""
        self.logger.info("Initializing Amazon Recommender System...")
        
        try:
            self.spark = create_spark_session()
            self.data_loader = DataLoader(self.spark)
            
            # Load data
            self.products_df, self.reviews_df = self.data_loader.load_data()
            
            # Initialize components
            self.search_engine = SearchEngine(self.products_df, self.reviews_df)
            self.recommender = HybridRecommender(self.spark, self.products_df, self.reviews_df)
            self.collab_filter = CollaborativeFiltering(self.spark, self.products_df, self.reviews_df)
            
            # Train collaborative filtering model
            self.logger.info("Training collaborative filtering model...")
            self.collab_filter.train_model()
            
            self.logger.info("Amazon Recommender System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            if self.spark:
                self.shutdown()
            raise
    
    def execute_search_query(self, query_type: str, **kwargs):
        """Execute search query with enhanced functionality"""
        return self.search_engine.search(query_type, **kwargs)
    
    def get_recommendations(self, user_id: str = None, product_id: str = None, n: int = 10):
        """Get recommendations using hybrid approach"""
        if user_id:
            return self.recommender.hybrid_recommendations(user_id, n)
        elif product_id:
            return self.recommender.similar_products(product_id, n)
        else:
            return self.recommender.popularity_based_recommendations(n=n)
    
    def get_collaborative_recommendations(self, user_id: str = None, product_id: str = None, n: int = 10):
        """Get recommendations using collaborative filtering"""
        if user_id:
            return self.collab_filter.recommend_for_user(user_id, n)
        elif product_id:
            return self.collab_filter.get_similar_products(product_id, n)
        else:
            return self.collab_filter.get_popular_products(n)
    
    def get_also_bought(self, product_id: str, n: int = 10):
        """Get 'customers who bought this also bought' recommendations"""
        return self.collab_filter.get_also_bought_products(product_id, n)
    
    def get_category_stats(self):
        """Get category statistics"""
        return self.search_engine.get_category_stats()
    
    def get_product_details(self, product_id: str):
        """Get detailed product information"""
        return self.search_engine.get_product_info(product_id)
    
    def get_user_history(self, user_id: str):
        """Get user's review history with category information"""
        return self.search_engine.get_user_review_history(user_id)
    
    def get_system_info(self):
        """Get system information"""
        product_count = self.products_df.count()
        review_count = self.reviews_df.count()
        user_count = self.reviews_df.select("customer_id").distinct().count()
        category_count = self.products_df.select("category").distinct().count()
        
        return {
            "products": product_count,
            "reviews": review_count,
            "users": user_count,
            "categories": category_count
        }
    
    def get_copurchasers_count(self, user_id: str, product_id: str):
        """Get number of customers purchasing the same product given user_id and product_id"""
        return self.search_engine.get_copurchasers_count(user_id, product_id)
    
    def shutdown(self):
        """Shutdown application"""
        if self.spark:
            stop_spark_session(self.spark)
        self.logger.info("Application shutdown completed")

def main():
    """Main function"""
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    app = AmazonRecommenderApp()
    
    try:
        app.initialize_data()
        
        info = app.get_system_info()
        print("\n" + "="*60)
        print("Amazon Recommender System - Ready!")
        print("="*60)
        print(f"Products loaded: {info['products']:,}")
        print(f"Reviews loaded: {info['reviews']:,}")
        print(f"Unique users: {info['users']:,}")
        print(f"Categories: {info['categories']:,}")
        print("\nSystem is ready for queries!")
        
        import time
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app.shutdown()

if __name__ == "__main__":
    main()