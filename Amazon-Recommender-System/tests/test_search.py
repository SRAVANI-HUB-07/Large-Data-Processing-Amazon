import pytest
from pyspark.sql import SparkSession, DataFrame
from src.search_engine.search import SearchEngine
from src.search_engine.queries import SearchQueries

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestSearchEngine") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture
def sample_data(spark_session):
    products_data = [
        (1, "B001", "Test Book 1", "Book", 100, ["B002", "B003"], 2, 50, 50, 4.5, []),
        (2, "B002", "Test Book 2", "Book", 200, ["B001", "B003"], 1, 30, 30, 4.2, []),
        (3, "D001", "Test DVD 1", "DVD", 50, ["D002"], 1, 100, 100, 4.8, []),
        (4, "D002", "Test DVD 2", "DVD", 150, ["D001"], 1, 80, 80, 4.0, []),
        (5, "B003", "Test Book 3", "Book", 300, ["B001", "B002"], 1, 10, 10, 3.5, [])
    ]
    
    reviews_data = [
        ("B001", "U001", 5, "2020-01-01", 10, 8),
        ("B001", "U002", 4, "2020-01-02", 8, 6),
        ("B002", "U001", 3, "2020-01-03", 5, 3),
        ("D001", "U003", 5, "2020-01-04", 15, 12),
        ("D001", "U001", 4, "2020-01-05", 12, 9),
        ("D002", "U002", 2, "2020-01-06", 3, 1)
    ]
    
    products_df = spark_session.createDataFrame(products_data, [
        "product_id", "asin", "title", "group", "salesrank", 
        "similar", "categories_count", "reviews_total", 
        "reviews_downloaded", "avg_rating", "reviews"
    ])
    
    reviews_df = spark_session.createDataFrame(reviews_data, [
        "asin", "customer_id", "rating", "review_date", "votes", "helpful"
    ])
    
    return products_df, reviews_df

def test_search_engine_initialization(sample_data):
    products_df, reviews_df = sample_data
    search_engine = SearchEngine(products_df, reviews_df)
    
    assert search_engine is not None
    assert search_engine.products_df.count() == 5
    assert search_engine.reviews_df.count() == 6

def test_best_sellers_query(sample_data):
    products_df, reviews_df = sample_data
    queries = SearchQueries()
    
    results = queries.best_sellers(products_df, "Book", 2)
    results_list = results.collect()
    
    assert len(results_list) == 2
    assert results_list[0]["asin"] == "B001"
    assert results_list[1]["asin"] == "B002"