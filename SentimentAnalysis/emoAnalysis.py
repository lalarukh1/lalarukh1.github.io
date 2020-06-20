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
import nltk
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
from textblob.classifiers import NaiveBayesClassifier as NBC
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import emoji
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.preprocessing import LabelEncoder
import sys
import time
import datetime
import cmath

data2019 = pd.read_csv('/Users/lalarukh/Desktop/Tweets/tweets-2019-all.csv')
data2018 = pd.read_csv('/Users/lalarukh/Desktop/Tweets/tweets-2018-all.csv')
data2017 = pd.read_csv('/Users/lalarukh/Desktop/Tweets/tweets-2017-all.csv')
# #  ---------------------------------------------------------------------
# negative_tweets_2019 = twitter_samples.strings('tweets-2019/negTextblob.json')
# positive_tweets_2019 = twitter_samples.strings('tweets-2019/posTextblob.json')
# positive_tweets_2018 = twitter_samples.strings('tweets-2018/posTextblob.json')
# negative_tweets_2018 = twitter_samples.strings('tweets-2018/negTextblob.json')
# negative_tweets_2017 = twitter_samples.strings('tweets-2017/negTextblob.json')
# positive_tweets_2017 = twitter_samples.strings('tweets-2017/posTextblob.json')
# all_negative_tweets = twitter_samples.strings('allNegativeTweets.json')
# all_positive_tweets = twitter_samples.strings('allPositiveTweets.json')
# # ----------------------------------------------------------------------------
#
def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def read_datasets(fname, t_type):
    data = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        data.append([line, t_type])
        line = f.readline()
    f.close()
    return data


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


def classify_dataset(data):
    return classifier.classify(extract_features(nltk.word_tokenize(data)))

start_time = time.time()

# read in joy , disgust, sadness, anger, fear training dataset
joy_feel = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/joy.txt', 'joy')
disgust_feel = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/disgust.txt', 'disgust')
sadness_feel = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/sadness.txt', 'sadness')
anger_feel = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/anger.txt', 'anger')
fear_feel = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/fear.txt', 'fear')

data = []
for (words, sentiment) in joy_feel + disgust_feel + sadness_feel + anger_feel + fear_feel:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    data.append((words_filtered, sentiment))

word_features = get_word_features(get_words_in_dataset(data))

training_set = nltk.classify.util.apply_features(extract_features, data)
classifier = NaiveBayesClassifier.train(training_set)

total = []
for i in range(5):
    total.append(5)

test_data = read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/joy.txt', 'joy')
total[0] = len(test_data)
test_data.extend(read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/sadness.txt', 'sadness'))
total[1] = len(test_data) - total[0]
test_data.extend(read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/disgust.txt', 'disgust'))
total[2] = len(test_data) - total[0] - total[1]
test_data.extend(read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/anger.txt', 'anger'))
total[3] = len(test_data) - total[0] - total[1] - total[2]
test_data.extend(read_datasets('/Users/lalarukh/PycharmProjects/SentimentAnalysis/venv/Scripts/fear.txt', 'fear'))
total[4] = len(test_data) - total[0] - total[1] - total[2] - total[3]

accuracy = []

for i in range(5):
    accuracy.append(0)

emo = ["joy", "disgust", "sadness", "anger", "fear"]
num = [0, 1, 2, 3, 4]

emdict = dict(zip(emo, num))

prec = [0, 0, 0, 0, 0]

for data in test_data:
    result = classify_dataset(data[0])
    if result == "joy" and result == data[1]:
        accuracy[0] += 1
    else:
        prec[emdict[result]] += 1
    if result == "disgust" and result == data[1]:
        accuracy[1] += 1
    else:
        prec[emdict[result]] += 1
    if result == "sadness" and result == data[1]:
        accuracy[2] += 1
    else:
        prec[emdict[result]] += 1
    if result == "anger" and result == data[1]:
        accuracy[3] += 1
    else:
        prec[emdict[result]] += 1
    if result == "fear" and result == data[1]:
        accuracy[4] += 1
    else:
        prec[emdict[result]] += 1

tota = 0
tot = 0

for acu in accuracy:
    tota += acu
for t in total:
    tot += t

print('Total accuracy: %f%% (%d/20).' % (tota / tot * 100, tota))
print('Total accuracy - joy: %f%% (%d/20).' % (accuracy[0] / total[0] * 100, accuracy[0]))
print('Total accuracy - disgust: %f%% (%d/20).' % (accuracy[1] / total[1] * 100, accuracy[1]))
print('Total accuracy - sadness %f%% (%d/20).' % (accuracy[2] / total[2] * 100, accuracy[2]))
print('Total accuracy - anger %f%% (%d/20).' % (accuracy[3] / total[3] * 100, accuracy[3]))
print('Total accuracy - fear %f%% (%d/20).' % (accuracy[4] / total[4] * 100, accuracy[4]))

i = 0
joy = 0
disgust = 0
sadness = 0
anger = 0
fear = 0
for x in data2017['text']:
    if pd.isnull(data2017['text'][i]):
        print('Null')
        i = i + 1
    else:
        emotion = classify_dataset(x)
        i = i + 1
        if emotion == 'joy':
            joy = joy + 1
        if emotion == 'disgust':
            disgust = disgust + 1
        if emotion == 'sadness':
            sadness = sadness + 1
        if emotion == 'anger':
            anger = anger + 1
        if emotion == 'fear':
            fear = fear + 1
        if i == 10000:
            print('10000th turn')
        if i == 50000:
            print('50000th turn')
        if i == 100000:
            print('100000th turn - close!')

print('Total 2017:', i)
print('Joy:', joy)
print('Disgust:', disgust)
print('Sadness:', sadness)
print('Anger:', anger)
print('Fear:', fear)

elapsed_time = time.time() - start_time
print("processing time:", elapsed_time, "seconds")

joy19 = 31268
disgust19 = 17461
sadness19 = 14895
anger19 = 9111
fear19 = 3148

joy18 = 31837
disgust18 = 16121
sadness18 = 13632
anger18 = 7339
fear18 = 2659

joy17 = 42170
disgust17 = 16955
sadness17 = 19241
anger17 = 4775
fear17 = 2459

TotalTweets19 = len(data2019)
joyPercentage19 = joy19*100 / TotalTweets19
disgustPercentage19 = disgust19*100 / TotalTweets19
sadnessPercentage19 = sadness19*100 / TotalTweets19
angerPercentage19 = anger19*100 / TotalTweets19
fearPercentage19 = fear19*100 / TotalTweets19

TotalTweets18 = len(data2018)
joyPercentage18 = joy18*100 / TotalTweets18
disgustPercentage18 = disgust18*100 / TotalTweets18
sadnessPercentage18 = sadness18*100 / TotalTweets18
angerPercentage18 = anger18*100 / TotalTweets18
fearPercentage18 = fear18*100 / TotalTweets18

TotalTweets17 = len(data2017)
joyPercentage17 = joy17*100 / TotalTweets17
disgustPercentage17 = disgust17*100 / TotalTweets17
sadnessPercentage17 = sadness17*100 / TotalTweets17
angerPercentage17 = anger17*100 / TotalTweets17
fearPercentage17 = fear17*100 / TotalTweets17

df = pd.DataFrame([
    ['Joy', '2019', joyPercentage19], ['Joy', '2018', joyPercentage18], ['Joy', '2017', joyPercentage17],
    ['Disgust', '2019', disgustPercentage19], ['Disgust', '2018', disgustPercentage18], ['Disgust', '2017', disgustPercentage17],
    ['Sadness', '2019', sadnessPercentage19], ['Sadness', '2018', sadnessPercentage18], ['Sadness', '2017', sadnessPercentage17],
    ['Anger', '2019', angerPercentage19], ['Anger', '2018', angerPercentage18], ['Anger', '2017', angerPercentage17],
    ['Fear', '2019', fearPercentage19], ['Fear', '2018', fearPercentage18], ['Fear', '2017', fearPercentage17]
],
    columns=['Emotion', 'year', 'val'])
ax = df.pivot("Emotion", "year", "val").plot(kind='bar')

plt.suptitle('Percentage Emotion Distribution Over the Years')
plt.show()

Pie chart 2019
labels = 'JOY', 'DISGUST', 'SADNESS', 'ANGER', 'FEAR'
sizes = [31268, 17461, 14895, 9111, 3148]
fig1, ax1 = plt.subplots()
explode = [0, 0, 0, 0, 0]
colors = ['#fa5457', '#fa8925', '#f6d51f', '#01b4bc', '#5fa55a']
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
ax1.axis('equal')
plt.tight_layout()
plt.suptitle('Emotion Analysis for 2019')
plt.show()

# Pie chart 2018
labels = 'JOY', 'DISGUST', 'SADNESS', 'ANGER', 'FEAR'
sizes = [31837, 16121, 13632, 7339, 2659]
colors = ['#fa5457', '#fa8925', '#f6d51f', '#01b4bc', '#5fa55a']
fig1, ax1 = plt.subplots()
explode = [0, 0, 0, 0, 0]
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
ax1.axis('equal')
plt.tight_layout()
plt.suptitle('Emotion Analysis for 2018')
plt.show()


# Pie chart 2017
labels = 'JOY', 'DISGUST', 'SADNESS', 'ANGER', 'FEAR'
sizes = [42170, 16955, 19241, 4775, 2459]
colors = ['#fa5457', '#fa8925', '#f6d51f', '#01b4bc', '#5fa55a']
fig1, ax1 = plt.subplots()
explode = [0, 0, 0, 0, 0]
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=colors)
ax1.axis('equal')
plt.tight_layout()
plt.suptitle('Emotion Analysis for 2017')
plt.show()
