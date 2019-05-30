from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


def main(context):
    """Main function takes a Spark SQL context."""
    labeled_df = spark.read.option("header", "true").csv("labeled_data.csv")
    labeled_df.write.parquet("labeledParquet")

    submission_df = spark.read.option('compression', 'gzip').json("submissions.json.bz2")
    comments_df = spark.read.option('compression', 'gzip').json("comments-minimal.json.bz2")

    submission_df.write.parquet("submissionParquet")
    comments_df.write.parquet("commentsParquet")

    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
