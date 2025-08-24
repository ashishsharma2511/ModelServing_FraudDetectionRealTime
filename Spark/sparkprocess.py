import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, hour, struct, to_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType,IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler

# ----------------------
# Spark session setup
# ----------------------
spark = SparkSession.builder \
    .appName("frddetect") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ----------------------
# Kafka JSON schema
# ----------------------
schema = StructType([
    StructField("transaction_id", IntegerType()),
    StructField("user_id", IntegerType()),
    StructField("amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("category", StringType()),
    StructField("time_of_day", IntegerType()),
    StructField("device_type", StringType()),
    StructField("location", StringType())
])


mlflow.set_tracking_uri("http://127.0.0.1:5000")
pipeline_model = mlflow.spark.load_model("runs:/ba57ac2e3e7841c98fa83330f6264ffe/model")

# ----------------------
# Load streaming data from Kafka
# ----------------------
kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "evtransactions")
    .option("startingOffsets", "latest")
    .load()
)

# ----------------------
# Parse JSON
# ----------------------
json_df = kafka_df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), schema).alias("data")) \
    .select("data.*")

# merchant_indexer = StringIndexer(inputCol="merchant", outputCol="merchantIndex")
# category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
# device_type_indexer = StringIndexer(inputCol="device_type", outputCol="deviceTypeIndex")
# location_indexer = StringIndexer(inputCol="location", outputCol="locationIndex")

# json_df = merchant_indexer.fit(json_df).transform(json_df)
# json_df = category_indexer.fit(json_df).transform(json_df)
# json_df = device_type_indexer.fit(json_df).transform(json_df)
# json_df = location_indexer.fit(json_df).transform(json_df)

# feature_cols= ["amount","merchantIndex","categoryIndex","time_of_day","deviceTypeIndex","locationIndex"]
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# json_df = assembler.transform(json_df)

# ----------------------
# Make predictions
# ----------------------
predictions_df = pipeline_model.transform(json_df)

# ----------------------
# Output predictions to Kafka
# ----------------------
kafka_out = predictions_df.select(
    to_json(struct("amount","merchant","category","time_of_day","deviceType","location", "prediction")).alias("value")
)

kafka_query = kafka_out.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "churn_predictions") \
    .option("checkpointLocation", "/tmp/spark_checkpoints/churn_predict/kafka") \
    .outputMode("append") \
    .start()


# ----------------------
# Wait for streams to finish
# -------
kafka_query.awaitTermination()


