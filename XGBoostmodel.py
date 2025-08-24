import mlflow
import mlflow.spark
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

#SparkSession
SparkSession = SparkSession.builder.appName("XGBoostModel").getOrCreate()

#read data
df = SparkSession.read.csv("Training_data.csv", header=True, inferSchema=True)
merchant_indexer = StringIndexer(inputCol="merchant", outputCol="merchantIndex")
category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
device_type_indexer = StringIndexer(inputCol="device_type", outputCol="deviceTypeIndex")
location_indexer = StringIndexer(inputCol="location", outputCol="locationIndex")

df = merchant_indexer.fit(df).transform(df)
df = category_indexer.fit(df).transform(df)
df = device_type_indexer.fit(df).transform(df)
df = location_indexer.fit(df).transform(df)
df = df.withColumn("is_fraud", col("is_fraud").cast("int"))

feature_cols= ["amount","merchantIndex","categoryIndex","time_of_day","deviceTypeIndex","locationIndex"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("XGModelforFraudDetection")

with mlflow.start_run():
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="is_fraud",
        prediction_col="prediction",
        numWorkers=2
    )

    # Train -> Model
    model = xgb.fit(train_df)
    mlflow.spark.log_model(model, "model")

    #predict
    predictions = model.transform(test_df)

    #evaluate model
    evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.spark.log_model(model, "xgboost_classifier")

SparkSession.stop()