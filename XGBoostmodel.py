import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

#SparkSession
SparkSession = SparkSession.builder.appName("XGBoostModel").getOrCreate()

#read data
df = SparkSession.read.csv("Training_data.csv", header=True, inferSchema=True)

feature_cols= ["amount","merchant","category","time_of_day","device_type","location"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("XGModelforFraudDetection")

with mlflow.start_run():
    model = SparkXGBClassifierModel()
    model.fit(train_df)
    mlflow.spark.log_model(model, "model")

    #predict
    predictions = model.transform(test_df)

    #log metrics
    mlflow.spark.log_metrics(predictions, "predictions")

    #evaluate model
    evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.spark.log_model(model, "xgboost_classifier")

SparkSession.stop()