
import os
import pandas as pd
import numpy as np
#import nltk
from nltk.corpus import stopwords
import nltk.corpus as nc
import string
import datetime
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#from nltk.stem.porter import PorterStemmer

TRAINING_NEW = False
workDir = "C:\\Users\\rounayak\\Desktop\\PPG"
os.chdir(workDir)

stopwords_english = stopwords.words('english') + list(string.punctuation) + [i.lower() for i in nc.names.words()]
#porter = PorterStemmer()

def clear_text(text_input):
    try:
        clear_text = [i.lower() for i in text_input.split() if (i not in stopwords_english and i==i)]
        #clear_text = [porter.stem(i.lower()) for i in text_input.split() if (i not in stopwords_english and i==i)]
        #clear_text = [i.lower() for i in text_input.split() if (i not in stopwords_english and i==i and i.isdigit()==False)]
        return " ".join(clear_text)
    except Exception as e:
        print(e)
        return ""

def get_file_name(input_str):
    return datetime.datetime.now().strftime("%Y-%m-%d%H%M%S")+input_str

Sentiments_table = pd.read_excel("NLP_Lexicon_Final.xlsx")
#,usecols=['word','sentiment'])
Sentiments_table = Sentiments_table[['word','sentiment']]
Sentiments_table = Sentiments_table[~pd.isnull(Sentiments_table.word)]
Sentiments_table.drop_duplicates(subset='word',inplace=True)

inFileName = "NLP_Master_Dataset.csv"
COLNAMES = ['customer_id','territory_id','call_type','call_reason',
            'call_subject','call_result','start_time']


os.chdir(workDir+"\\week_3_task")

TEXT_COL = 'TEXT_COL_CLEANED'

inData = pd.read_csv(inFileName)
print(inData.shape)

inData.drop_duplicates(subset=['customer_id', 'contact_id',TEXT_COL],inplace=True)

inData = inData[~pd.isnull(inData[TEXT_COL])]

if TRAINING_NEW == True:
    lda = LatentDirichletAllocation(n_components=10, max_iter=50,
                                    learning_method='online',
                                    learning_offset=10.,
                                    n_jobs=-1,
                                    verbose=5,
                                    batch_size=50000,
                                    random_state=0)
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=100,
                                    ngram_range=(1, 4),
                                    stop_words='english')
    
    tf = tf_vectorizer.fit_transform(inData[TEXT_COL])
    lda.fit(tf)
    pickle.dump(tf_vectorizer,open(get_file_name("tf_vectorizer.pkl"),"wb")) 
    pickle.dump(lda,open(get_file_name("lda.pkl"),"wb"))
    
else:
    tf_vectorizer = pickle.load(open(".\\upto2018\\2018-05-15181343tf_vectorizer.pkl","rb")) 
    lda = pickle.load(open(".\\upto2018\\2018-05-15181503lda.pkl","rb"))
    print("loaded pickles")
    out_topics = lda.transform(tf_vectorizer.transform(inData.loc[:,'TEXT_COL_CLEANED']))
    inData.loc[:,'TopicID'] = out_topics.argmax(axis=1)
    inData.to_csv("TopicSelected.csv",index=False)

tf_feature_names = tf_vectorizer.get_feature_names()

Normalized_Component = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
Normalized_Component = Normalized_Component/Normalized_Component.sum(axis=0)
Normalized_Component = Normalized_Component/ Normalized_Component.sum(axis=1)[:, np.newaxis]

outDF = pd.DataFrame()

TOP_N_WORDS = 12
for Topic_ID,lda_topic_x_vector in enumerate(lda.components_):
    word_indices = lda_topic_x_vector.argsort()[::-1][:TOP_N_WORDS]
    df = pd.DataFrame.from_dict({tf_feature_names[i]:lda_topic_x_vector[i] for i in word_indices},orient='index')
    df.index.name = 'word'
    df.columns = ['score']
    df.loc[:,'TopicID'] = 'TopicID_'+str(Topic_ID+1)
    df.reset_index(inplace=True)
    outDF = pd.concat([outDF,df],axis=0,ignore_index=True)
    #print("\n")

print(outDF)
outDF.to_csv("Word_Topic_Distribution.csv",index=False)

