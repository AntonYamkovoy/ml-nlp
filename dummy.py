from sklearn.dummy import DummyClassifier
import jsonlines
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np

X = []
y = []
z = []

with jsonlines.open('data/reviews.txt', 'r') as f:
    for item in f:
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

X = np.array(X)
y = np.array(y)
z = np.array(z)

train_scores = []
test_scores = []

kfolds = KFold(n_splits=5, random_state=None, shuffle=True)
kfolds.get_n_splits(X)

for train_index, test_index in kfolds.split(X):
    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = z[train_index], z[test_index]
    clf = DummyClassifier(strategy = 'stratified').fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_pred_train, y_train))
    print(classification_report(y_pred_test, y_test))
    rep_train = classification_report(y_pred_train, y_train, output_dict=True)
    rep_test = classification_report(y_pred_test, y_test,  output_dict=True)

    train_scores.append(rep_train['macro avg']['f1-score'])

    test_scores.append(rep_test['macro avg']['f1-score'])


print(np.average(train_scores))
print(np.average(test_scores))
