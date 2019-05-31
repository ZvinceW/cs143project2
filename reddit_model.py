from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf

import re
import string
import argparse
import json

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
        new_word = None;
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

    # Run if parquets aren't available
    # labeled_df = spark.read.option("header", "true").csv("labeled_data.csv")
    # labeled_df.write.parquet("labeledParquet")
    #
    # submission_df = spark.read.option('compression', 'gzip').json("submissions.json.bz2")
    # comments_df = spark.read.option('compression', 'gzip').json("comments-minimal.json.bz2")
    #
    # submission_df.write.parquet("submissionParquet")
    # comments_df.write.parquet("commentsParquet")

    # Run if parquets are available
    labeled_df = sqlContext.read.parquet('./labeledParquet/')
    comments_df = sqlContext.read.parquet('./commentsParquet/')
    submission_df = sqlContext.read.parquet('./submissionParquet/')

    # TASK 2
    complete_df = labeled_df.join(comments_df, labeled_df.Input_id == comments_df.id)

    # TASK 4
    sanitize_udf = udf(lambda comment: sanitize(comment), ArrayType(StringType()))

    complete_df = complete_df.withColumn("grams", sanitize_udf(complete_df["body"]))

    # TASK 5
    combine_udf = udf(lambda grams: combine(grams), ArrayType(StringType()))

    complete_df = complete_df.withColumn("flat_grams", combine_udf(complete_df["grams"]))

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
