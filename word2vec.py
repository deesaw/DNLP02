import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
spell = SpellChecker()
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
def speak(text):
    try:
        a=TextBlob(text).correct()
    except:
        a=text
    finally:
        return(str(a))
df = pd.read_csv('ICD_CODES.tsv', sep='\t')
df['Issue']=df['CODES'].str.split(n=1).str[1]
df['CODES'] = df.CODES.str.split().str.get(0)
df['Issuew2v']=df['Issue']
df['Issuetextblob']=df['Issue']
df['misspelledSpellChecker']=df['Issue']
df['CorrectSpelling']=df['Issue']
df['CorrectSpellingCandidates']=df['Issue']
df['Stemming']=df['Issue']
df['Lemmatization']=df['Issue']
for i in range(len(df['Issue'])):
    df['misspelledSpellChecker'][i]=[]
    df['CorrectSpelling'][i]=[]
    df['CorrectSpellingCandidates'][i]=[]
    df['Stemming'][i]=[]
    df['Lemmatization'][i]=[]
    text=df['Issue'][i]
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = text.lower()
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    df['Issuew2v'][i]=nltk.word_tokenize(text)
    df['Stemming'][i]=[ps.stem(word)  for word in df['Issuew2v'][i] ]
    df['Lemmatization'][i]=[wordnet.lemmatize(word)  for word in df['Issuew2v'][i] ]
    df['Issuew2v'][i]=[word for word in df['Issuew2v'][i] if word not in stopwords.words('english')]
    df['Issuetextblob'][i]=speak(text)
    misspelled = spell.unknown(nltk.word_tokenize(text))
    df['misspelledSpellChecker'][i]=misspelled
    df['CorrectSpelling'][i]=[spell.correction(word) for word in df['misspelledSpellChecker'][i]]
    df['CorrectSpellingCandidates'][i]=[spell.candidates(word) for word in df['misspelledSpellChecker'][i]]
model = Word2Vec(df['Issuew2v'], min_count=1)  
words = model.wv.vocab 
similar = model.wv.most_similar('fever')
