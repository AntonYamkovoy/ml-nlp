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
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

X = []
y = []
z = []

with jsonlines.open('data/reviews.txt', 'r') as f:
    for item in f:
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

def detect_text(text):
    try:
        language = detect(text)
    except:
        language = "unknown"

    return language


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
        'dutch', 'arabic', 'bulgarian',
        'danish', 'finnish', 'indonesian', 'vietnamese' ]

stop_words = []
for l in lang:
    for w in get_stop_words(l):
        stop_words.append(w)

def preprocess_text(text):
    detected_lang = detect_text(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and token != ' ' and token.strip() not in punctuation and len(token) > 1]
    stemmed = stem(tokens, detected_lang)
    if stemmed == None:
        text = ' '.join(tokens)
    else:
        text = ' '.join(stemmed)

    return text


processed_text = []
for x in X:
    processed_text.append(preprocess_text(x))

X_train_og, X_test_og, y_train_og, y_test_og = train_test_split(processed_text, z, test_size = 0.2)

def feature_selection(min_freq, max_freq, n_gram, train, test):
    vectorizer = TfidfVectorizer(ngram_range=n_gram, max_df = max_freq, min_df = min_freq, use_idf=True, sublinear_tf = True, smooth_idf= True)
    train_vec = vectorizer.fit_transform(train)
    test_vec = vectorizer.transform(test)

    return train_vec, test_vec


def train_model(train_vec, test_vec, y_train, y_test):

    #clf = linear_model.LogisticRegression(C = 1)
    #clf = DecisionTreeClassifier(random_state=0)
    clf = svm.SVC(kernel = 'linear', C = 10)
    clf.fit(train_vec, y_train)
    y_pred_test = clf.predict(test_vec)
    y_pred_train = clf.predict(train_vec)
    print(classification_report(y_train, y_pred_train))
    print(classification_report(y_test, y_pred_test))
    return classification_report(y_train, y_pred_train, output_dict=True), classification_report(y_test, y_pred_test, output_dict=True)
    #return classification_report(y_train, y_pred_train), classification_report(y_test, y_pred_test)


processed_text = np.array(processed_text)
X = np.array(X)
y = np.array(y)
z = np.array(z)




train_scores = []
test_scores = []


kfolds = KFold(n_splits=5, random_state=None, shuffle=True)
kfolds.get_n_splits(processed_text)

for train_index, test_index in kfolds.split(processed_text):
    X_train, X_test = processed_text[train_index], processed_text[test_index]

    y_train, y_test = y[train_index], y[test_index]
    train_vec, test_vec = feature_selection(2,200,(1,1), X_train, X_test)
    rep_train, rep_test = train_model(train_vec, test_vec, y_train, y_test)
    train_scores.append(rep_train['macro avg']['f1-score'])

    test_scores.append(rep_test['macro avg']['f1-score'])


print(np.average(train_scores))
print(np.average(test_scores))
