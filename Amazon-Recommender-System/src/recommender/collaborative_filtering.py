from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    def __init__(self, spark: SparkSession, max_iter: int = 10, reg_param: float = 0.1, 
                 rank: int = 10, cold_start_strategy: str = "drop"):
        self.spark = spark
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.cold_start_strategy = cold_start_strategy
        self.model = None
        self.user_indexer = None
        self.item_indexer = None
        
    def prepare_data(self, reviews_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        logger.info("Preparing data for collaborative filtering")
        
        from pyspark.ml.feature import StringIndexer
        
        self.user_indexer = StringIndexer(inputCol="customer_id", outputCol="user_index")
        self.item_indexer = StringIndexer(inputCol="asin", outputCol="item_index")
        
        indexed_df = self.user_indexer.fit(reviews_df).transform(reviews_df)
        indexed_df = self.item_indexer.fit(indexed_df).transform(indexed_df)
        
        indexed_df.cache()
        
        (training, test) = indexed_df.randomSplit([0.8, 0.2])
        
        logger.info(f"Training set: {training.count()}, Test set: {test.count()}")
        
        return indexed_df, training, test
    
    def train(self, training_data: DataFrame, hyperparameter_tuning: bool = False) -> None:
        logger.info("Training ALS model")
        
        if hyperparameter_tuning:
            self._train_with_tuning(training_data)
        else:
            self._train_simple(training_data)
    
    def _train_simple(self, training_data: DataFrame) -> None:
        als = ALS(maxIter=self.max_iter,
                 regParam=self.reg_param,
                 rank=self.rank,
                 userCol="user_index",
                 itemCol="item_index",
                 ratingCol="rating",
                 coldStartStrategy=self.cold_start_strategy,
                 implicitPrefs=False)
        
        self.model = als.fit(training_data)
        logger.info("ALS model training completed")
    
    def _train_with_tuning(self, training_data: DataFrame) -> None:
        als = ALS(userCol="user_index",
                 itemCol="item_index",
                 ratingCol="rating",
                 coldStartStrategy=self.cold_start_strategy,
                 implicitPrefs=False)
        
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5, 10, 15]) \
            .addGrid(als.maxIter, [5, 10]) \
            .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
            .build()
        
        evaluator = RegressionEvaluator(metricName="rmse", 
                                      labelCol="rating", 
                                      predictionCol="prediction")
        
        cv = CrossValidator(estimator=als,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3,
                          parallelism=2)
        
        cv_model = cv.fit(training_data)
        self.model = cv_model.bestModel
        
        logger.info(f"Best model parameters: {cv_model.bestModel.extractParamMap()}")
    
    def evaluate(self, test_data: DataFrame) -> Dict[str, float]:
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.transform(test_data)
        predictions = predictions.dropna()
        
        evaluator_rmse = RegressionEvaluator(metricName="rmse", 
                                           labelCol="rating", 
                                           predictionCol="prediction")
        evaluator_mae = RegressionEvaluator(metricName="mae", 
                                          labelCol="rating", 
                                          predictionCol="prediction")
        
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "test_samples": test_data.count(),
            "prediction_samples": predictions.count()
        }
        
        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return metrics
    
    def recommend_for_user(self, user_id: str, n: int = 10, 
                          products_df: DataFrame = None) -> DataFrame:
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        user_df = self.spark.createDataFrame([(user_id,)], ["customer_id"])
        indexed_user = self.user_indexer.transform(user_df)
        
        if indexed_user.isEmpty():
            logger.warning(f"User {user_id} not found in training data")
            return self.spark.createDataFrame([], self.spark.sparkContext.emptyRDD())
        
        user_index = indexed_user.first()["user_index"]
        
        user_subset = self.spark.createDataFrame([(user_index,)], ["user_index"])
        recommendations = self.model.recommendForUserSubset(user_subset, n)
        
        from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, FloatType
        
        exploded_recs = recommendations.select(
            F.explode("recommendations").alias("rec")
        ).select(
            "rec.item_index", "rec.rating"
        )
        
        return exploded_recs
    
    def recommend_for_all_users(self, n: int = 10) -> DataFrame:
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.recommendForAllUsers(n)
    
    def save_model(self, path: str):
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.write().overwrite().save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        from pyspark.ml.recommendation import ALSModel
        self.model = ALSModel.load(path)
        logger.info(f"Model loaded from {path}")