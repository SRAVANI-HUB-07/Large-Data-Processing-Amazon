import pytest
from pyspark.sql import SparkSession
from src.recommender.algorithms import HybridRecommender, RecommendationEvaluator

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestRecommender") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture
def sample_recommendation_data(spark_session):
    products_data = [
        (1, "B001", "Book 1", "Book", 100, [], 1, 50, 50, 4.5),
        (2, "B002", "Book 2", "Book", 200, [], 1, 30, 30, 4.2),
        (3, "D001", "DVD 1", "DVD", 50, [], 1, 100, 100, 4.8),
        (4, "M001", "Music 1", "Music", 150, [], 1, 80, 80, 4.0)
    ]
    
    reviews_data = [
        ("B001", "U001", 5, "2020-01-01", 10, 8),
        ("B001", "U002", 4, "2020-01-02", 8, 6),
        ("B002", "U001", 3, "2020-01-03", 5, 3),
        ("D001", "U003", 5, "2020-01-04", 15, 12),
        ("M001", "U001", 4, "2020-01-05", 12, 9)
    ]
    
    products_df = spark_session.createDataFrame(products_data, [
        "product_id", "asin", "title", "group", "salesrank", 
        "similar", "categories_count", "reviews_total", 
        "reviews_downloaded", "avg_rating"
    ])
    
    reviews_df = spark_session.createDataFrame(reviews_data, [
        "asin", "customer_id", "rating", "review_date", "votes", "helpful"
    ])
    
    return products_df, reviews_df

def test_hybrid_recommender_initialization(spark_session, sample_recommendation_data):
    products_df, reviews_df = sample_recommendation_data
    recommender = HybridRecommender(spark_session, products_df, reviews_df)
    
    assert recommender is not None
    assert recommender.products_df.count() == 4
    assert recommender.reviews_df.count() == 5

def test_popularity_based_recommendations(spark_session, sample_recommendation_data):
    products_df, reviews_df = sample_recommendation_data
    recommender = HybridRecommender(spark_session, products_df, reviews_df)
    
    recommendations = recommender.popularity_based_recommendations(n=2)
    recommendations_list = recommendations.collect()
    
    assert len(recommendations_list) == 2
    assert recommendations_list[0]["asin"] == "D001"