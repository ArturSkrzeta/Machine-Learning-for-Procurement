import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('unspsc2.csv', engine='python')
df = shuffle(df)

x = df['unspsc']    #features
y = df['team']      #labels

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(['computer','transport'])

for pred in predictions:
    print(pred)

# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))
