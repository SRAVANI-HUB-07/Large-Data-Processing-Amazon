import json
import gzip
import logging
from typing import Dict, List, Any, Iterator, Optional
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, MapType

logger = logging.getLogger(__name__)

class AmazonJSONParser:
    """
    Robust parser for Amazon product metadata in JSON format
    Handles malformed JSON and various data formats
    """
    
    def parse_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Parse Amazon JSON metadata file with robust error handling
        """
        logger.info(f"Starting to parse JSON file: {file_path}")
        
        # Handle gzipped files
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'
        
        successful_parses = 0
        failed_parses = 0
        
        with open_func(file_path, mode, encoding='utf-8', errors='ignore') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                product = self._safe_parse_line(line, line_num)
                if product:
                    successful_parses += 1
                    yield product
                else:
                    failed_parses += 1
                    
                    # Log only occasional failures to avoid spam
                    if failed_parses <= 10 or failed_parses % 100 == 0:
                        logger.warning(f"Failed to parse line {line_num} (total failures: {failed_parses})")
        
        logger.info(f"JSON parsing completed: {successful_parses} successful, {failed_parses} failed")
        
        if successful_parses == 0:
            logger.error("No products successfully parsed from the file")
            raise Exception("No valid JSON products found in the dataset")
    
    def _safe_parse_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """
        Safely parse a single line with comprehensive error handling
        """
        try:
            # Try direct JSON parsing first
            product_data = json.loads(line)
            return self._transform_product(product_data)
            
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed_line = self._fix_json_issues(line)
            if fixed_line != line:
                try:
                    product_data = json.loads(fixed_line)
                    return self._transform_product(product_data)
                except:
                    pass
            
            # If still failing, skip this line
            return None
            
        except Exception as e:
            # Handle other parsing errors
            logger.debug(f"Unexpected error parsing line {line_num}: {e}")
            return None
    
    def _fix_json_issues(self, line: str) -> str:
        """
        Attempt to fix common JSON formatting issues
        """
        fixed = line
        
        # Fix single quotes to double quotes
        fixed = fixed.replace("'", '"')
        
        # Fix unescaped quotes
        fixed = fixed.replace('\\"', '"')
        
        # Remove trailing commas
        fixed = fixed.replace(',}', '}').replace(',]', ']')
        
        # Fix missing quotes around property names
        import re
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        
        return fixed
    
    def _transform_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform JSON product to our standard format with validation
        """
        try:
            # Basic validation - must have ASIN
            asin = product.get('asin')
            if not asin or not isinstance(asin, str):
                return None
            
            transformed = {
                'product_id': asin,
                'asin': asin,
                'title': str(product.get('title', ''))[:500],  # Limit title length
                'group': self._extract_category(product),
                'salesrank': self._safe_int(product.get('salesRank', 999999)),
                'similar': self._extract_similar_products(product),
                'categories_count': self._count_categories(product),
                'reviews_total': 0,
                'reviews_downloaded': 0,
                'avg_rating': 0.0,
                'reviews': []
            }
            
            # Extract and validate rating
            rating = product.get('rating')
            if rating and isinstance(rating, (int, float)):
                transformed['avg_rating'] = max(0.0, min(5.0, float(rating)))
            
            # Extract and transform reviews
            reviews = self._extract_reviews(product)
            if reviews:
                transformed['reviews'] = reviews
                transformed['reviews_total'] = len(reviews)
                transformed['reviews_downloaded'] = len(reviews)
                
                # Calculate average rating from reviews if not provided
                if transformed['avg_rating'] == 0.0:
                    valid_ratings = [r['rating'] for r in reviews if 1 <= r['rating'] <= 5]
                    if valid_ratings:
                        transformed['avg_rating'] = sum(valid_ratings) / len(valid_ratings)
            
            return transformed
            
        except Exception as e:
            logger.debug(f"Error transforming product: {e}")
            return None
    
    def _extract_category(self, product: Dict[str, Any]) -> str:
        """Extract category from product data"""
        categories = product.get('categories', [])
        if categories and isinstance(categories, list) and categories[0]:
            first_category = categories[0]
            if isinstance(first_category, list) and first_category:
                return str(first_category[0])[:100]
        return 'General'
    
    def _extract_similar_products(self, product: Dict[str, Any]) -> List[str]:
        """Extract similar products with validation"""
        similar = []
        related = product.get('related', {})
        
        if isinstance(related, dict):
            for key in ['also_bought', 'also_viewed', 'bought_together']:
                products = related.get(key, [])
                if isinstance(products, list):
                    similar.extend([str(p) for p in products if p][:5])
        
        return list(set(similar))[:10]  # Remove duplicates and limit
    
    def _count_categories(self, product: Dict[str, Any]) -> int:
        """Count categories with validation"""
        categories = product.get('categories', [])
        if isinstance(categories, list):
            return len(categories)
        return 0
    
    def _extract_reviews(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and validate reviews"""
        reviews = []
        review_data = product.get('reviews')
        
        if not isinstance(review_data, list):
            return reviews
        
        for review in review_data:
            if not isinstance(review, dict):
                continue
                
            try:
                customer_id = review.get('reviewerID', '')
                rating = self._safe_int(review.get('overall', 0))
                
                # Only include reviews with valid customer ID and rating
                if customer_id and 1 <= rating <= 5:
                    reviews.append({
                        'date': str(review.get('reviewTime', ''))[:20],
                        'customer_id': str(customer_id)[:50],
                        'rating': rating,
                        'votes': self._safe_int(review.get('vote', 0)),
                        'helpful': self._extract_helpful(review.get('helpful'))
                    })
            except Exception:
                continue
        
        return reviews
    
    def _extract_helpful(self, helpful_data: Any) -> int:
        """Extract helpful votes from various formats"""
        if isinstance(helpful_data, list) and len(helpful_data) >= 1:
            return self._safe_int(helpful_data[0])
        elif isinstance(helpful_data, int):
            return helpful_data
        return 0
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert to integer"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

def create_json_product_schema():
    """Define Spark schema for JSON product data"""
    return StructType([
        StructField("product_id", StringType(), True),
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