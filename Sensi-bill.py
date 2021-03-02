import numpy as np
import pandas as pd
import re
import nltk 
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv('ICD_CODES.tsv', sep='\t')
df['Issue']=df['CODES'].str.split(n=1).str[1]
df['CODES'] = df.CODES.str.split().str.get(0)
print(df.head())
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
X = df['Issue']
y = df['CODES']

spell = SpellChecker()
sente=df['Issue']
# find those words that may be misspelled
for s in sente:
    misspelled = spell.unknown(nltk.word_tokenize(s))
    for word in misspelled:
        print(spell.correction(word))
        print(spell.candidates(word))