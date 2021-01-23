
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from langdetect import detect
from sklearn import linear_model
import jsonlines
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.metrics import classification_report
import random
random.seed(4)

X = []
y = []
z = []

with jsonlines.open('data/reviews_big.txt', 'r') as f:
    for item in f:
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])



def createWordList(line):
    wordList2 =[]
    wordList1 = line.split()
    for word in wordList1:
        cleanWord = ""
        for char in word:
            if char in '!,.?":;-_ \`()':
                char = ""
            cleanWord += char
        if cleanWord != '':
            wordList2.append(cleanWord.strip())
    return wordList2

def detect_text(text):
    try:
        language = detect(text)
    except:
        language = "unknown"

    return language


def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]

def calcScore(word, X, y): # mode == true for voted_up
    rating = 0
    for i in range(len(X)):
        words =  word_tokenize(X[i])
        if word in words:
            cond = y[i]
            if cond:
                rating += 1
            else:
                rating += -1
    return rating

def stem(tokens, lang):
    stemmer = None
    if lang == 'en':
        stemmer = SnowballStemmer('english')
    elif lang == 'ru':
        stemmer = SnowballStemmer('russian')
    elif lang == 'de':
        stemmer = SnowballStemmer('german')
    elif lang == 'pt':
        stemmer = SnowballStemmer('portuguese')
    elif lang == 'es':
        stemmer = SnowballStemmer('spanish')
    if stemmer is not None:
        return [stemmer.stem(t) for t in tokens]
    else:
        return None

stop_words = []

lang = ['english', 'russian', 'portuguese',
             'turkish', 'spanish', 'swedish',
             'polish', 'french', 'german',
             'dutch', 'arabic']

stop_words = []
for l in lang:
    for w in get_stop_words(l):
        stop_words.append(w)



def preprocess_text(text):
    detected_lang = detect_text(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and token != ' ' and token.strip() not in punctuation]
    stemmed = stem(tokens, detected_lang)
    if stemmed == None:
        text = ' '.join(tokens)
    else:
        text = ' '.join(stemmed)

    return text



processed_text = []
for x in X:
    processed_text.append(preprocess_text(x))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("Finished split")
print("Percentage Voted Up Training : ", y_train.count(True)/len(y_train))
print("Percentage Voted Up Test : ", y_test.count(True)/len(y_test))


''''
vec = CountVectorizer(min_df = 10, max_df = 50).fit(X_train)
bag_of_words = vec.transform(X_train)
sum_words = bag_of_words.sum(axis=0)
words = [word for word, idx in vec.vocabulary_.items()]
print(words)

with open('data/lex_words_train', 'wb') as filehandle:
    pickle.dump(words, filehandle)

'''

'''
words = []
with open('data/lex_words_train', 'rb') as filehandle:
    words = pickle.load(filehandle)

word_ratings = []
print(len(words))

for word in words:
    rating = calcScore(word, X_train, y_train)
    print(rating, ' ', word)
    word_ratings.append((word, rating))

test_pred = []

with open('data/lex_ratings_beta', 'wb') as filehandle:
    pickle.dump(word_ratings, filehandle)

'''
'''
s = sum(j for i, j in word_ratings)
normed_word_ratings = []
for tuple in word_ratings:
    normed_word_ratings.append((tuple[0], (tuple[1]/s)))
'''

#print("Normalized: ", normed_word_ratings)


#limit = -4
limit = -4

word_ratings = []

with open('data/lex_ratings', 'rb') as filehandle:
    # read the data as binary data stream
  word_ratings = pickle.load(filehandle)


train_pred = []

for i in range(len(X_train)):
    train_word_list = createWordList(X_train[i])
    score = 0
    for w in train_word_list:
        for tuple in word_ratings:
            if w == tuple[0]:
                score += tuple[1]
    if score > limit:
        train_pred.append((True, y_train[i]))
    else:
        train_pred.append((False, y_train[i]))


y_true_train = [x for x,y in train_pred]
y_pred_train = [y for x,y in train_pred]


test_pred = []


for i in range(len(X_test)):
    test_word_list = createWordList(X_test[i])
    score = 0
    for w in test_word_list:
        for tuple in word_ratings:
            if w == tuple[0]:
                score += tuple[1]
    if score > limit:
        test_pred.append((True, y_test[i]))
    else:
        test_pred.append((False, y_test[i]))



y_true_test= [x for x,y in test_pred]
y_pred_test = [y for x,y in test_pred]


print(classification_report(y_true_train, y_pred_train))
print(classification_report(y_true_test, y_pred_test))
