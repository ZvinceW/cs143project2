#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import json


__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

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

    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    with open('sample.json') as json_file:
        samples = list(json_file)

        for sample in samples:
            entry = json.loads(sample)
            text = entry['body']
            print('ORIGINAL')
            print(text)

            print('')
            print('CLEAN')
            print(sanitize(text))
            print('')

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.
