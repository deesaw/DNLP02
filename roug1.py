import numpy as np
import pandas as pd
df = pd.read_csv('ICD_Test.tsv', sep='\t')
#df['Description']=df['Description'].str.lower()
#print(df.head())
#print(df.isnull().sum())
# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print(len(blanks))
df.dropna(inplace=True)
#print(df['label'].value_counts())

from sklearn.model_selection import train_test_split

X = df['Description']
y = df['ICD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train) 
# Form a prediction set
predictions = text_clf.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


