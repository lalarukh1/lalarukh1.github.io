
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import lxml
import selenium
from bs4 import BeautifulSoup
import base64
import sys
import jsonpickle
import os
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import seaborn as sns
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples
import csv, json
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict, Counter
from textblob.classifiers import NaiveBayesClassifier
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions

i=1
sentiment_scores = []
data2019 = pd.read_csv('C:/Users/lalar/Desktop/tweets-2019/tweets-2019-all.csv')
data2018 = pd.read_csv('C:/Users/lalar/Desktop/tweets-2018/tweets-2018-all.csv')
data2017 = pd.read_csv('C:/Users/lalar/Desktop/tweets-2017/tweets-2017-all.csv')

i=0
for x in data2019['text']:
    i = i + 1

print(i)

i=0
for x in data2018['text']:
    i = i + 1

print(i)

i=0
for x in data2017['text']:
    i = i + 1

print(i)

# preprocessing data for VADER

def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    return tweet

# -------------------------------------------------------
# Writing positive/negative tweets CSVs using textblob

# testing preprocessing method
for x in data2019['text']:
    print(x)
    print(clean_tweets(x))


with open('C:/Users/lalar/Desktop/tweets-2019/posVADER.csv', mode='w', encoding="utf-8") as pos_file, open('C:/Users/lalar/Desktop/tweets-2019/negVADER.csv', mode='w', encoding="utf-8") as neg_file:
    fieldnames = ['text']

    writer1 = csv.DictWriter(pos_file, fieldnames=fieldnames)
    writer2 = csv.DictWriter(neg_file, fieldnames=fieldnames)
    writer1.writeheader()
    writer2.writeheader()

    for x in data2019['text']:
        if pd.isnull(data2019['text'][i]):
            print('Null')
            i = i + 1
        else:
                    tweet = clean_tweets(data2019['text'][i])
                    analyser = SentimentIntensityAnalyzer()
                    sentiment = analyser.polarity_scores(tweet)
                    score = sentiment['compound']
                    print(score)
                    sentiment_scores.append(score)
                    if score >= 0.05:
                        writer1.writerow({'text': (tweet)})
                    elif score <= -0.05:
                        writer2.writerow({'text': (tweet)})
                    else:
                        print('neutral')
                        print(i)

        i = i + 1

#---------------------------------------------------------
# From CSV positive/negative tweets files to JSON files in Twitter_samples Data folder
#
with open('C:/Users/lalar/Desktop/tweets-2019/posVADER.csv', encoding="utf-8") as pos_file:
    posjsonfile = open('C:/Users/lalar/AppData/Roaming/nltk_data/corpora/twitter_samples/tweets-2019/posVADER.json', 'w')
    fieldnames = ['text']
    reader = csv.DictReader(pos_file)
    for row in reader:
        json.dump(row, posjsonfile)
        posjsonfile.write('\n')
#
with open('C:/Users/lalar/Desktop/tweets-2019/negVADER.csv', encoding="utf-8") as neg_file:
    negjsonfile = open('C:/Users/lalar/AppData/Roaming/nltk_data/corpora/twitter_samples/tweets-2019/negVADER.json', 'w')
    fieldnames = ['text']
    reader = csv.DictReader(neg_file)
    for row in reader:
        json.dump(row, negjsonfile)
        negjsonfile.write('\n')
#  ---------------------------------------------------------------------
negative_tweets_2019 = twitter_samples.strings('tweets-2019/negVADER.json')
positive_tweets_2019 = twitter_samples.strings('tweets-2019/posVADER.json')
positive_tweets_2018 = twitter_samples.strings('tweets-2018/posVADER.json')
negative_tweets_2018 = twitter_samples.strings('tweets-2018/negVADER.json')
negative_tweets_2017 = twitter_samples.strings('tweets-2017/negVADER.json')
positive_tweets_2017 = twitter_samples.strings('tweets-2017/posVADER.json')
all_negative_tweets = twitter_samples.strings('allNegativeTweetsVADER.json')
all_positive_tweets = twitter_samples.strings('allPositiveTweetsVADER.json')
# ----------------------------------------------------------------------------

# More preprocessing using json files
# #
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

print(lemmatize_sentence(tweet_tokens[0]))

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

stop_words = stopwords.words('english')

print(remove_noise(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('allPositiveTweetsVADER.json')
negative_tweet_tokens = twitter_samples.tokenized('allNegativeTweetsVADER.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:80203]
test_data = dataset[80203:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

positiveNB2019 = 0;
negativeNB2019 = 0;
positiveNB2017 = 0;
negativeNB2017 = 0;
positiveNB2018 = 0;
negativeNB2018 = 0;

for x in data2019['text']:
    custom_tokens = remove_noise(word_tokenize(str(x)))
    classification = classifier.classify(dict([token, True] for token in custom_tokens))
    if(classification == 'Positive'):
        positiveNB2019 = positiveNB2019 + 1
    elif(classification == 'Negative'):
        negativeNB2019 = negativeNB2019 + 1
    else:
        print('neutral')

print("Positive tweets 2019:", positiveNB2019)
print("Negative tweets 2019", negativeNB2019)

for x in data2018['text']:
    custom_tokens = remove_noise(word_tokenize(str(x)))
    classification = classifier.classify(dict([token, True] for token in custom_tokens))
    if(classification == 'Positive'):
        positiveNB2018 = positiveNB2018 + 1
    elif(classification == 'Negative'):
        negativeNB2018 = negativeNB2018 + 1
    else:
        print('neutral')

print("Positive tweets 2018:", positiveNB2018)
print("Negative tweets 2018", negativeNB2018)


for x in data2017['text']:
    custom_tokens = remove_noise(word_tokenize(str(x)))
    classification = classifier.classify(dict([token, True] for token in custom_tokens))
    if(classification == 'Positive'):
        positiveNB2017 = positiveNB2017 + 1
    elif(classification == 'Negative'):
        negativeNB2017 = negativeNB2017 + 1
    else:
        print('neutral')

print("Positive tweets 2017:", positiveNB2017)
print("Negative tweets 2017", negativeNB2017)

# ---------------------------------------------
# Graph for VADER Tweets

TotalTweets2019 = len(data2019)
NTweets2019 = len(negative_tweets_2019)
NTweets2019Percentage = NTweets2019*100 / TotalTweets2019
PTweets2019 = len(positive_tweets_2019)
PTweets2019Percentage = PTweets2019*100 / TotalTweets2019
TotalTweets2018 = len(data2018)
NTweets2018 = len(negative_tweets_2018)
NTweets2018Percentage = NTweets2018*100 / TotalTweets2018
PTweets2018 = len(positive_tweets_2018)
PTweets2018Percentage = PTweets2018*100 / TotalTweets2018
TotalTweets2017 = len(data2017)
NTweets2017 = len(negative_tweets_2017)
NTweets2017Percentage = NTweets2017*100 / TotalTweets2017
PTweets2017 = len(positive_tweets_2017)
PTweets2017Percentage = PTweets2017*100 / TotalTweets2017

df = pd.DataFrame([['Negative','2019',NTweets2019Percentage],['Negative','2018',NTweets2018Percentage],['Negative','2017',NTweets2017Percentage],['Positive','2019',PTweets2019Percentage],
                   ['Positive','2018',PTweets2018Percentage],['Positive','2017',PTweets2017Percentage]],columns=['Sentiment','Year','val'])

df.pivot("Sentiment", "Year", "val").plot(kind='bar')

plt.show()

# ---------------------------------------------------------------
# Graph for Naive Bayes Tweets

TotalTweets2019 = len(data2019)
NNBTweets2019 = 20630
NNBTweets2019Percentage = NNBTweets2019*100 / TotalTweets2019
PNBTweets2019 = 55270
PNBTweets2019Percentage = PNBTweets2019*100 / TotalTweets2019
TotalTweets2018 = len(data2018)
NNBTweets2018 = 16373
NNBTweets2018Percentage = NNBTweets2018*100 / TotalTweets2018
PNBTweets2018 = 55227
PNBTweets2018Percentage = PNBTweets2018*100 / TotalTweets2018
TotalTweets2017 = len(data2017)
NNBTweets2017 = 22927
NNBTweets2017Percentage = NNBTweets2017*100 / TotalTweets2017
PNBTweets2017 = 62674
PNBTweets2017Percentage = PNBTweets2017*100 / TotalTweets2017

df = pd.DataFrame([['Negative','2019',NNBTweets2019Percentage],['Negative','2018',NNBTweets2018Percentage],['Negative','2017',NNBTweets2017Percentage],['Positive','2019',PNBTweets2019Percentage],
                   ['Positive','2018',PNBTweets2018Percentage],['Positive','2017',PNBTweets2017Percentage]],columns=['polarity','year','val'])

df.pivot("polarity", "year", "val").plot(kind='bar')

plt.show()
