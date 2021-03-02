import pandas as pd,numpy as np
from tqdm import tqdm
df=pd.read_csv('ICD_Test.tsv', sep='\t')



df=df.replace(np.nan,'',regex=True)
X = df['Description']
y=df['ICD'].values

import re
import nltk
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
for i in tqdm(range(len(df))):
    review = re.sub('[^a-zA-Z0-9]',' ',df['Description'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
 
from sklearn.feature_extraction.text import CountVectorizer    
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(corpus).toarray()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)   


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.01)

from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=150,criterion='entropy')
classifier1.fit(X_train,y_train)
predRF=classifier1.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))

YY=le.inverse_transform(classifier1.predict(X_train))
XX=cv.inverse_transform(X_train)
print('Training Data-Accuracy score: {}'.format(accuracy_score(le.inverse_transform(y_train), YY)))

df_train = pd.DataFrame()
df_train['INPUT']=XX
df_train['OUTPUT']=YY
