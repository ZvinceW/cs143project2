from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer

import re
import string
import argparse
import json

states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
          "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
          "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
          "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
          "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
          "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
          "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
          "Wyoming"]

def is_state(text):
    return text in states

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:

    # 1. replace newlines and tabs with single space
    text = re.sub(r"[\n\t]", " ", text)
    # 2. remove urls
    text = re.sub(r"][\(]?http\S+[\)]?|][\(]\S+[\)]", "", text)
    text = re.sub(r"\[", "", text)
    text = re.sub(r"\S+\.\S+\.\S+", "", text)
    # 5. split on space (text should become an array of shorter strings)
    text = text.split(" ")
    nospace_text = []
    for word in text:
        if word != "":
            nospace_text.append(word)
    text = nospace_text
    # print(text)
    # 6&7. separate external punctuations + remove punctuation
    good_punc = {".", "!", "?", ",", ";", ":"}
    do_not_remove = {".", "!", "?", ",", ";", ":", "(", ")", "'", "\"", "-", "--"}

    temp_text = []
    for word in text:
        last_char = word[len(word) - 1]
        if last_char in good_punc:
            word = word[:len(word) - 1]
            temp_text.append(word)
            temp_text.append(last_char)
            # print(word)
        else:
            temp_text.append(word)
    text = temp_text

    temp = []
    for word in text:
        new_word = None
        for character in word:
            if (character not in do_not_remove and not character.isalnum()):
                new_word = word.replace(character, "")
        if (new_word is not None):
            temp.append(new_word)
        else:
            temp.append(word)

    text = temp

    # 8. convert all to lowercase
    text = [word.lower() for word in text]

    # 10.
    parsed_text = ' '.join(text)

    # unigram
    no_punctuation = []
    for word in text:
        if (word not in good_punc):
            no_punctuation.append(word)

    unigrams = ' '.join(no_punctuation)

    bigrams = []
    for i in range(0, len(no_punctuation)-1):
        bigram = '_'.join([no_punctuation[i], no_punctuation[i+1]])
        bigrams.append(bigram)

    bigrams = ' '.join(bigrams)

    trigrams = []
    for i in range(0, len(no_punctuation)-2):
        trigram = '_'.join([no_punctuation[i], no_punctuation[i+1], no_punctuation[i+2]])
        trigrams.append(trigram)

    trigrams = ' '.join(trigrams)

    return [unigrams, bigrams, trigrams]

def combine(grams):
    split = [gram.split() for gram in grams]
    return [word for gram in split for word in gram]

def main(context):
    """Main function takes a Spark SQL context."""
    # TASK 1

    # Run if parquets aren't available, this creates the data frames based of the data inputs
    # labeled_df = spark.read.option("header", "true").csv("labeled_data.csv")
    # labeled_df.write.parquet("labeledParquet")

    # submission_df = spark.read.option('compression', 'gzip').json("submissions.json.bz2")
    # comments_df = spark.read.option('compression', 'gzip').json("comments-minimal.json.bz2")

    # submission_df.write.parquet("submissionParquet")
    # comments_df.write.parquet("commentsParquet")

    # Run if parquets are available
    # This imports data from parquets instead of the actual file so it's faster
    labeled_df = sqlContext.read.parquet('./labeledParquet/')
    comments_df = sqlContext.read.parquet('./commentsParquet/')
    submission_df = sqlContext.read.parquet('./submissionParquet/')

    # TASK 2
    complete_df = labeled_df.join(comments_df, labeled_df.Input_id == comments_df.id).select("Input_id","labeldem","labelgop","labeldjt","body")

    # TASK 4
    # Creates Spark UDF that will run sanitize on every comment
    sanitize_udf = udf(lambda comment: sanitize(comment), ArrayType(StringType()))

    # Creates a new column grams that contains the output of sanitize
    complete_df = complete_df.withColumn("grams", sanitize_udf(complete_df["body"]))

    # TASK 5
    # Flattens the grams column so that it's just a list of strings instead of list of lists
    combine_udf = udf(lambda grams: combine(grams), ArrayType(StringType()))

    complete_df = complete_df.withColumn("flat_grams", combine_udf(complete_df["grams"]))

    # TASK 6A

    # fit a CountVectorizerModel from the corpus.
    # source: https://spark.apache.org/docs/latest/ml-features.html#countvectorizer
    cv = CountVectorizer(inputCol="flat_grams", outputCol="features", minDF=10.0, binary = True)
    model = cv.fit(complete_df)
    result = model.transform(complete_df)

    # TASK 6B
    positive_udf = udf(lambda label: 1 if label == "1" else 0, IntegerType())
    negative_udf = udf(lambda label: 1 if label == "-1" else 0, IntegerType())
    result = result.withColumn("poslabel", positive_udf(result["labeldjt"]))
    result = result.withColumn("neglabel", negative_udf(result["labeldjt"]))

    # TASK 7
    # Bunch of imports (may need more)
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10)
    neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator(labelCol="poslabel")
    negEvaluator = BinaryClassificationEvaluator(labelCol="neglabel")
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = result.randomSplit([0.5, 0.5])
    negTrain, negTest = result.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")

    # TASK 8
    cut_link_udf = udf(lambda link: link[3:], StringType())
    comments_df = comments_df.withColumn("cut_link_id", cut_link_udf(comments_df["link_id"]))
    t8_df = comments_df.join(submission_df, comments_df.cut_link_id == submission_df.id).select(comments_df.id, comments_df.created_utc, submission_df.title, comments_df.author_flair_text.alias("state"), comments_df.body)
    # may need a udf to sort out null and non-state flairs

    # TASK 9
    # source: https://stackoverflow.com/questions/45838698/pyspark-dataframe-filter-on-multiple-columns
    from pyspark.sql.functions import col
    t9_df = t8_df.filter(~col("body").like("%\s%") & ~col("body").like("&gt%"))
    t9_df = t9_df.withColumn("grams", sanitize_udf(t9_df["body"]))
    t9_df = t9_df.withColumn("flat_grams", combine_udf(t9_df["grams"]))

    result = model.transform(t9_df)

    posResult = posModel.transform(result)

    pos_udf = udf(lambda prob: 1 if prob[1] > 0.2 else 0)
    neg_udf = udf(lambda prob: 1 if prob[1] > 0.25 else 0)

    # source: https://stackoverflow.com/questions/29600673/how-to-delete-columns-in-pyspark-dataframe
    t9_pos_df = posResult.withColumn("pos", pos_udf(posResult["probability"]))
    t9_pos_df = t9_pos_df.drop("probability", "rawPrediction", "prediction")

    negResult = negModel.transform(t9_pos_df)

    t9_pos_neg_df = negResult.withColumn("neg", neg_udf(negResult["probability"]))
    t9_pos_neg_df = t9_pos_neg_df.drop("probability", "rawPrediction", "prediction", "grams", "flat_grams", "features")

    # this takes two hours dont accidentally run
    # t9_pos_neg_df.write.parquet("t9_final.parquet")

    # TASK 10

    t9_pos_neg_df.createOrReplaceTempView("temp_view")

    percentage_df = sqlContext.sql("""
        SELECT comment_id,
        AVG(pos),
        AVG(neg)
        FROM temp_view
        GROUP BY comment_id""")

    timeseries_df = sqlContext.sql("""
        SELECT FROM_UNIXTIME(created_utc, '%Y%M%D') AS date,
        AVG(pos) AS positive,
        AVG(neg) AS negative
        FROM temp_view
        GROUP BY date""")

    sqlContext.registerFunction("state_udf", is_state, BooleanType())

    state_percentage_df = sqlContext.sql("""
        SELECT author_flair_text AS state,
        AVG(pos) AS positvite,
        AVG(neg) AS negative
        FROM temp_view
        WHERE(is_state_udf(author_flair_text))
        GROUP BY state")""")

    comment_score_df = sqlContext.sql("""
        SELECT comments_score as comment_score,
        AVG(pos_label) AS Positive,
        AVG(neg_label) AS Negative
        FROM predicted
        GROUP BY comments_score
    """)

    submission_score_df = sqlContext.sql("""
        SELECT submission_score,
        AVG(pos_label) AS Positive,
        AVG(neg_label) AS Negative
        FROM predicted
        GROUP BY submission_score
    """)

    percentage_df.orderBy("AVG(pos)", ascending=False).limit(10).repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("top_10_pos.csv")
    percentage_df.orderBy("AVG(neg)", ascending=False).limit(10).repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("top_10_neg.csv")
    timeseries_df.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("time_data.csv")
    state_percentage_df.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("state_data.csv")
    percent_comment_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("comment_score.csv")
    percent_submission_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("submission_score.csv")




if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
