import pytest
import tempfile
import os
from pyspark.sql import SparkSession
from src.data_processing.parser import AmazonDataParser

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestAmazonParser") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture
def sample_data_file():
    sample_data = """Id:   1
ASIN: 0827229534
  title: Patterns of Preaching: A Sermon Sampler
  group: Book
  salesrank: 396585
  similar: 5 0804215715 156101074X 0687023955 0687074231 082721619X
  categories: 2
  reviews: total: 2  downloaded: 2  avg rating: 5
    2000-7-28  cutomer: A2JW67OY8U6HHK  rating: 5  votes:  10  helpful:   9
    2000-8-29  cutomer: A2JW67OY8U6HHK  rating: 5  votes:  12  helpful:  10

Id:   2
ASIN: 0738700983
  title: Heart of the Storm: Book 1 of the Dark Crescent
  group: Book
  salesrank: 396585
  similar: 5 0738700983 0738700983 0738700983 0738700983 0738700983
  categories: 1
  reviews: total: 0  downloaded: 0  avg rating: 0"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_data)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

def test_parser_initialization():
    parser = AmazonDataParser()
    assert parser is not None
    assert hasattr(parser, 'patterns')

def test_parse_sample_data(sample_data_file):
    parser = AmazonDataParser()
    products = list(parser.parse_file(sample_data_file))
    
    assert len(products) == 2
    assert products[0]['product_id'] == 1
    assert products[0]['asin'] == '0827229534'
    assert products[0]['title'] == 'Patterns of Preaching: A Sermon Sampler'
    assert products[0]['group'] == 'Book'
    assert products[0]['salesrank'] == 396585
    assert len(products[0]['similar']) == 5
    assert products[0]['reviews_total'] == 2
    assert products[0]['avg_rating'] == 5.0