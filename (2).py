from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, BooleanType
import matplotlib.pyplot as plt
import numpy as np
import json

spark = SparkSession.builder.appName("Read Json").getOrCreate()
json = spark.read.format("json").option("inferSchema", "true").load("/media/hp/New Volume/All_Amazon_Review.json/amazon.json")
# Define schema for 'amazon' collection
amazon_schema = StructType([
    StructField("_id", StringType()),
    StructField("overall", IntegerType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", StringType())
])

# Create a SparkSession
#spark = SparkSession.builder.appName("Read Json").getOrCreate()
spark = SparkSession.builder \
    .appName("amazon") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/local.amazon") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/local.amazon") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.1.1") \
   # .getOrCreate()

# Read the 'amazon' collection from MongoDB into a PySpark DataFrame
df = spark.read \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .option("collection", "amazon") \
    .schema(amazon_schema) \
    .load()

# Extract specific columns from the DataFrame
overall_list = [x['overall'] for x in df.select('overall').limit(3000).collect()]
print(np.unique(overall_list, return_counts=True))

review_list = [y['reviewTime'] for y in df.select('reviewTime').limit(3000).collect()]
print(np.unique(review_list, return_counts=True))

verified_list = [z['verified'] for z in df.select('verified').limit(3000).collect()]

# Calculate correlation coefficient between review frequency and rating
corr = np.corrcoef(overall_list, range(len(overall_list)))[0, 1]
print(f"Pearson correlation coefficient between Reviews and Ratings is: {corr:.2f}")
print(f"We can conclude that there is a moderately negative correlation between review frequency and rating based on the Pearson correlation coefficient of {corr:.2f}. This indicates that the overall rating tends to decrease as the number of reviews increases.")

# Plot histogram of overall ratings
plt.hist(overall_list, bins=np.arange(0.5, 6.6, 1), color='gray', edgecolor='black')
plt.xticks(np.arange(1, 6))
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Overall Ratings')
plt.show()

# Plot histogram of review times
plt.hist(review_list, bins=30, color='gray', edgecolor='black')
plt.xlabel('Review Time')
plt.ylabel('Frequency')
plt.title('Time of Reviews')
plt.show()

# Stop the SparkSession
spark.stop()

