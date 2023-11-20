import os
import time
import findspark
import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
from IPython.display import display, clear_output
from pyspark.sql.streaming import DataStreamReader
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

# os.environ['PYSPARK_SUBMIT_ARGS'] = f'--packages org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION} pyspark-shell'

findspark.init()

SCALA_VERSION = '2.12'
SPARK_VERSION = '3.1.2'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
    'org.apache.kafka:kafka-clients:3.6.0'
]

spark = (SparkSession
         .builder
         .master("local")
         .appName('hsd-spark-kafka')
         .config("spark.jars.packages", ",".join(packages))         
         .getOrCreate())

timestampformat = "yyyy-MM-dd HH:mm:ss"
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

df = (spark.readStream.format('kafka')
      .option("kafka.bootstrap.servers", "localhost:9092") 
      .option("subscribe", "rawData") 
      .option("startingOffsets", "earliest")
      .option("failOnDataLoss", "false")
      .load())


schema_value = StructType(
    [StructField("author",StringType(),True),
    StructField("datetime",StringType(),True),
    StructField("raw_comment",StringType(),True),
    StructField("clean_comment",StringType(),True),
    StructField("label",IntegerType(),True)])
# -------------------------------***-----------------------------
df_json = (df
           .selectExpr("CAST(value AS STRING)")
           .withColumn("value",f.from_json("value",schema_value)))

df_column = (df_json.select(f.col("value.author").alias("user"),
                           f.to_timestamp(f.regexp_replace('value.datetime','[TZ]',' '),timestampformat).alias("timestamp"),
                           f.col("value.raw_comment").alias("raw_comment"),
                           f.col("value.clean_comment").alias("clean_comment"),
                           f.col("value.label").alias("label")
                               ))

df_count = (df_column.groupBy('label').agg(f.count('label').alias('count'))
            .withColumn('sentiment',f.when(df_column.label==1,'OFFENSIVE')
                        .when(df_column.label==0,'CLEAN')
                        .otherwise('HATE'))
           .select(f.col('sentiment'),f.col('count')))

df_statistic = df_column.groupBy('label') \
                       .agg(
                           f.sum(f.when(f.col('label') == 0, 1).otherwise(0)).alias('CleanCount'),
                           f.sum(f.when(f.col('label') == 1, 1).otherwise(0)).alias('OffensiveCount'),
                           f.sum(f.when(f.col('label') == 2, 1).otherwise(0)).alias('HateCount'))
'''
df_haters = (df_column.select('user','label')
            .where(df_column.label != 0)
            .groupBy('user')
            .agg(f.count('label').alias('most_hate_speech'))
            .orderBy('most_hate_speech',ascending=False)
            .withColumn("id",f.lit(1)))
'''
# ------------------------------ *** -----------------------------------
ds = (df_column
      .select(f.to_json(f.struct('user','timestamp',
                                    'raw_comment','clean_comment',
                                    'label')).alias('value'))
      .selectExpr("CAST(value AS STRING)") 
      .writeStream 
      .format("kafka") 
      .outputMode("append")
      .option("kafka.bootstrap.servers", "localhost:9092") 
      .option("topic", "cleanData") 
      .option("checkpointLocation","checkpoints/df_column")
      .start())

# ds_count = (df_count
#       .select(f.to_json(f.struct("sentiment","count")).alias("value"))
#       .writeStream \
#       .format("kafka") \
#       .option("kafka.bootstrap.servers", "localhost:9092") \
#       .option("topic", "cleanData") \
#       .option("checkpointLocation", "checkpoints/df_count") \
#       .outputMode("complete") \
#       .start()
# )

df_statistic = (df_statistic
      .select(f.to_json(f.struct('CleanCount', 'OffensiveCount', 'HateCount')).alias("value"))
      .writeStream \
      .format("kafka") \
      .option("kafka.bootstrap.servers", "localhost:9092") \
      .option("topic", "statistic") \
      .option("checkpointLocation", "checkpoints/df_count") \
      .outputMode("complete") \
      .start()
)


'''
# Show users commenting hate speech
ds_haters = (df_haters
            .select(f.to_json(f.struct('user','most_hate_speech')).alias('value'))
            .selectExpr("CAST(value AS STRING)")
            .writeStream
            .format("kafka")
            .option("kafka.bootstrap.servers", "localhost:9092") 
            .option("topic", "hsd") 
            .option("checkpointLocation", "home/david/checkpoints_haters")
            .outputMode("complete")
            .start())
'''
ds.awaitTermination()
