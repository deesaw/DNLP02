import numpy as np
import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
df = pd.read_csv('ICD_CODES.tsv', sep='\t')
df['Issue']=df['CODES'].str.split(n=1).str[1]
df['CODES'] = df.CODES.str.split().str.get(0)
print(df.head())


from sklearn.model_selection import train_test_split
X = df['Issue']
y = df['CODES']
'''
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
#sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(XX)):
    review = re.sub('[^a-zA-Z]', ' ', XX[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#X=pd.DataFrame (corpus,columns=['Description'])
X=corpus
y=pd.get_dummies(yy)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])


# Feed the training data through the pipeline
Xclf=text_clf.fit(X_train, y_train) 

# Form a prediction set
Y_predict=text_clf.predict(y_test)
