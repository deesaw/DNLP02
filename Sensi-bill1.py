import numpy as np
import pandas as pd
import re
import nltk 
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
spell = SpellChecker()
def speak(text):
#    while True:
    try:
        a=TextBlob(text).correct()
    except:
        a=text
    finally:
        print('Corrected:')
        print(a)
        return(str(a))
     
nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv('ICD_CODES.tsv', sep='\t')
df['Issue']=df['CODES'].str.split(n=1).str[1]
df['CODES'] = df.CODES.str.split().str.get(0)

from sklearn.model_selection import train_test_split
X = df['Issue']
y = df['CODES']

sente=df['Issue']
correct=[]
for s in sente:
    print('Actual:'+ s)
    correct.append(speak(s))
'''    misspelled = spell.unknown(nltk.word_tokenize(s))
    for word in misspelled:
        print(spell.correction(word))
        print(spell.candidates(word))

'''   