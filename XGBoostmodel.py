import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ----------------------
# Spark session
# ----------------------
spark = SparkSession.builder.appName("XGBoostModel").getOrCreate()

# ----------------------
# Read data
# ----------------------
df = spark.read.csv("Training_data.csv", header=True, inferSchema=True)
df = df.withColumn("is_fraud", col("is_fraud").cast("int"))

# ----------------------
# Preprocessing pipeline
# ----------------------
merchant_indexer = StringIndexer(inputCol="merchant", outputCol="merchantIndex")
category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
device_type_indexer = StringIndexer(inputCol="device_type", outputCol="deviceTypeIndex")
location_indexer = StringIndexer(inputCol="location", outputCol="locationIndex")
feature_cols = ["amount","merchantIndex","categoryIndex","time_of_day","deviceTypeIndex","locationIndex"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

preprocessing_pipeline = Pipeline(stages=[merchant_indexer, category_indexer, device_type_indexer, location_indexer, assembler])
preprocessing_model = preprocessing_pipeline.fit(df)
df_transformed = preprocessing_model.transform(df)

train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)

# ----------------------
# MLflow setup
# ----------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("XGModelforFraudDetection")

with mlflow.start_run():
    # ----------------------
    # Train XGBoost
    # ----------------------
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="is_fraud",
        prediction_col="prediction",
        numWorkers=2
    )
    xgb_model = xgb.fit(train_df)

    # ----------------------
    # Evaluate
    # ----------------------
    predictions = xgb_model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("accuracy", accuracy)

    # ----------------------
    # Log models separately
    # ----------------------
    mlflow.spark.log_model(preprocessing_model, "preprocessing_pipeline")
    mlflow.spark.log_model(xgb_model, "xgb_model")

spark.stop()
