import nltk 
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
import re
from re import search
import pandas as pd
from itertools import chain 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from PyDictionary import PyDictionary
dictionary=PyDictionary()

nltk.download('stopwords')
ps = PorterStemmer()
wordlem=WordNetLemmatizer()
all_stopwords = stopwords.words('english')
all_stopwords.append('treat')
all_stopwords.append('treated')

def synonyms(word):
    synonym = []
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym.append(l.name())
    if dictionary.synonym(word):
        for s in dictionary.synonym(word):
            synonym.append(s)
    synonym=set(synonym)
    print(synonym)
    return(synonym) 


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

df = pd.read_csv('ICD_Test.tsv', sep='\t')
df['Description']=df['Description'].str.lower()

def lookupsim(string):
    string=string.lower()
    soriginal=string
    sstring = re.sub('[^a-zA-Z0-9]', ' ', string)
    simtextinput=nltk.word_tokenize(sstring)
    synonymss=[]
    synonymss= [ synonyms(word) for word in simtextinput]
    flatten_list = list(chain.from_iterable(synonymss)) 
    my_set = set(flatten_list)
    flatten_list_u = list(my_set)
    simtextinputstem= [ ps.stem(word) for word in simtextinput]
#    simtextinputstemsynonyms= [ ps.stem(word) for word in flatten_list_u]
#    print(flatten_list_u)
    description=[]
    score=[]
    f=[]
    for el in df['Description']:
            el=el.lower()
            if(soriginal==el):
                print('orginal')
                return df['ICD'][df.Description == soriginal].values
                
            description.append(el)
            el=nltk.word_tokenize(el)
            f=[ps.stem(word) for word in el if word in flatten_list_u]
#            print(f)
            el=[ps.stem(word) for word in el if not word in all_stopwords]
            scor=jaccard_similarity(simtextinputstem,el)
#            print(scor)
            if scor!=0:
                print('Actual Search')
                score.append(scor)
                print(el, end=':')
                print(simtextinputstem, end=':')
                print(scor)
                
            else:
                scor=jaccard_similarity(f,el)
                print('Synonym search')
                score.append(scor)
                print(el, end=':')
                print(f, end=':')
                print(simtextinputstem, end=':')
                print(scor)
    print(description)        
    if soriginal in el:
        return df['ICD'][df.Description == soriginal].values
    elif string in df['Description']:
        return df['ICD'][df.Description == string].values
    elif max(score)> 0.20:
        return df['ICD'][score.index(max(score))]
    else:
        return"Not Found"
print(lookupsim('play'))   
 
 

'''
def aresynonyms(word1,word2):
    synonym = []
    for syn in wordnet.synsets(word1): 
        for l in syn.lemmas(): 
            synonym.append(l.name())
    if word2 in synonym:
        text="It is a Synonym"
    else:
        text="It is not a Synonym"
    return(text)

#b=aresynonyms('work','exploit')
#print(b)
     
def lookupsimlem(string):
    string=string.lower()
    sstring = re.sub('[^a-zA-Z0-9]', ' ', string)
    simtextinput=nltk.word_tokenize(sstring)
    simtextinputstem= [ wordlem.lemmatize(word) for word in simtextinput]
    score=[]
    for el in df['Description']:
            el=el.lower()
            el=nltk.word_tokenize(el)
            el=[wordlem.lemmatize(word) for word in el if not word in all_stopwords]
            scor=jaccard_similarity(simtextinputstem,el)
            score.append(scor)
            print(el, end=':')
            print(simtextinputstem, end=':')
            print(scor)
    if string in df['Description']:
        return df['ICD'][df.Description.lower() == string].values
    elif max(score)> 0.20:
        return df['ICD'][score.index(max(score))]
    else:
        return"Not Found"       
  
print(lookupsimlem('wwwwwwww'))   
print(lookupsimlem('Diabetic disorders'))     
'''


'''

def lookup(string):
    string=string.lower()
    simtextinput=nltk.word_tokenize(string)
    ssimtextinput=[ps.stem(word) for word in nltk.word_tokenize(string)]
    review = re.sub('[^a-zA-Z0-9]', ' ', string)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in all_stopwords]
    review = ' '.join(review)
    search=review
    lstring=wordlem.lemmatize(string)
    sstring=ps.stem(search) 
    print(string)
    print(search)
    print(lstring)
    print(sstring)
    l=[]
    aa=1
    listtokenize={}
    stemtokenize={}
    lemmtokenize={}
    for s in df['Description']:
        l.append(s)
        listtokenize[s]=nltk.word_tokenize(s)
        stemtokenize[s]=[ps.stem(word) for word in nltk.word_tokenize(s)]
        stemtokenize[s]=' '.join(stemtokenize[s])
        lemmtokenize[s]=[wordlem.lemmatize(word) for word in nltk.word_tokenize(s)]
        lemmtokenize[s]=' '.join(lemmtokenize[s])
    print(stemtokenize)
    print(l)
    if string in l:
        return df['ICD'][df.Description == string].values
    elif jaccard_similarity(ssimtextinput,['diabet','treat'])>50:
        return jaccard_similarity(ssimtextinput,['diabet','treat'])
    elif sstring in l:
        return df['ICD'][df.Description.str.contains(sstring)].values 
    elif lstring in l:
        return df['ICD'][df.Description.str.contains(lstring)].values
    elif aa==1:
        for key, value in stemtokenize.items():
            value=value.lower()
            print(value)
            print(key)
             
            if re.search(sstring, value):
                return df['ICD'][df.Description.str.contains(key)].values
            else:
                for key1, value1 in lemmtokenize.items():
                    if re.search(lstring, value):
                        return df['ICD'][df.Description.str.contains(key1)].values
                    else:
                        return "Not found"
    else:
        return "Not found"
            
'''   

