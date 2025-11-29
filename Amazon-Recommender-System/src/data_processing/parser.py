import re
import gzip
import logging
from typing import Dict, List, Any, Iterator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, MapType

logger = logging.getLogger(__name__)

class AmazonDataParser:
    def __init__(self):
        self.patterns = {
            'Id': re.compile(r'Id:\s+(\d+)'),
            'ASIN': re.compile(r'ASIN:\s+(\w+)'),
            'title': re.compile(r'title:\s+(.+)'),
            'group': re.compile(r'group:\s+(.+)'),
            'salesrank': re.compile(r'salesrank:\s+(\d+)'),
            'similar': re.compile(r'similar:\s+(\d+)\s+(.*)'),
            'categories': re.compile(r'categories:\s+(\d+)'),
            'reviews': re.compile(r'reviews:\s+total:\s+(\d+)\s+downloaded:\s+(\d+)\s+avg rating:\s+([\d.]+)'),
            'review': re.compile(r'(\d{4}-\d{1,2}-\d{1,2})\s+cutomer:\s+(\w+)\s+rating:\s+(\d+)\s+votes:\s+(\d+)\s+helpful:\s+(\d+)')
        }
    
    def parse_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        logger.info(f"Starting to parse file: {file_path}")
        
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'
        
        with open_func(file_path, mode, encoding='utf-8', errors='ignore') as file:
            current_product = {}
            in_reviews = False
            
            for line in file:
                line = line.strip()
                
                if not line:
                    if current_product:
                        yield current_product
                        current_product = {}
                        in_reviews = False
                    continue
                
                if line.startswith('Id:') and current_product:
                    yield current_product
                    current_product = {}
                    in_reviews = False
                
                self._parse_line(line, current_product)
                
                if 'reviews_total' in current_product and not in_reviews:
                    if current_product['reviews_total'] > 0:
                        in_reviews = True
                        current_product.setdefault('reviews', [])
            
            if current_product:
                yield current_product
        
        logger.info("File parsing completed")
    
    def _parse_line(self, line: str, product: Dict[str, Any]):
        for field, pattern in self.patterns.items():
            match = pattern.match(line)
            if match:
                if field == 'Id':
                    product['product_id'] = int(match.group(1))
                elif field == 'ASIN':
                    product['asin'] = match.group(1)
                elif field == 'title':
                    product['title'] = match.group(1)
                elif field == 'group':
                    product['group'] = match.group(1)
                elif field == 'salesrank':
                    product['salesrank'] = int(match.group(1))
                elif field == 'similar':
                    num_similar = int(match.group(1))
                    similar_products = match.group(2).split() if match.group(2) else []
                    product['similar'] = similar_products[:num_similar]
                elif field == 'categories':
                    product['categories_count'] = int(match.group(1))
                elif field == 'reviews':
                    product['reviews_total'] = int(match.group(1))
                    product['reviews_downloaded'] = int(match.group(2))
                    product['avg_rating'] = float(match.group(3))
                elif field == 'review' and 'reviews' in product:
                    review = {
                        'date': match.group(1),
                        'customer_id': match.group(2),
                        'rating': int(match.group(3)),
                        'votes': int(match.group(4)),
                        'helpful': int(match.group(5))
                    }
                    product['reviews'].append(review)
                break

def create_product_schema():
    return StructType([
        StructField("product_id", IntegerType(), True),
        StructField("asin", StringType(), True),
        StructField("title", StringType(), True),
        StructField("group", StringType(), True),
        StructField("salesrank", IntegerType(), True),
        StructField("similar", ArrayType(StringType()), True),
        StructField("categories_count", IntegerType(), True),
        StructField("reviews_total", IntegerType(), True),
        StructField("reviews_downloaded", IntegerType(), True),
        StructField("avg_rating", FloatType(), True),
        StructField("reviews", ArrayType(MapType(StringType(), StringType())), True)
    ]) 