
import os
import pandas as pd
import numpy as np
#import nltk
from nltk.corpus import stopwords
import nltk.corpus as nc
import string
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.stem.porter import PorterStemmer

workDir = "C:\\Users\\rounayak\\Desktop\\PPG\\"
os.chdir(workDir)

stopwords_english = stopwords.words('english') + list(string.punctuation) + [i.lower() for i in nc.names.words()]
#porter = PorterStemmer()

def clear_text(text_input):
    try:
        #clear_text = [porter.stem(i.lower()) for i in text_input.split() if (i not in stopwords_english and i==i)]
        clear_text = [i.lower() for i in text_input.split() if (i not in stopwords_english and i==i)]
        #clear_text = [i.lower() for i in text_input.split() if (i not in stopwords_english and i==i and i.isdigit()==False)]
        return " ".join(clear_text)
    except Exception as e:
        print(e)
        return ""

inFileName = "NLP Full Dataset_Prod.csv"
COLNAMES = ['customer_id','territory_id','call_type','call_reason',
            'call_subject','call_result','start_time']

TEXT_COL = 'concatenated_Text'

Sentiments_table = pd.read_excel("NLP_Lexicon_Final.xlsx")
#,usecols=['word','sentiment'])
Sentiments_table = Sentiments_table[['word','sentiment']]
Sentiments_table = Sentiments_table[~pd.isnull(Sentiments_table.word)]
Sentiments_table.drop_duplicates(subset='word',inplace=True)

inData = pd.read_csv(inFileName,parse_dates=['creation_date'])
print(inData[inData.creation_date>pd.Timestamp('2018-01-01 00:00:01')].shape)

inData.sample(n=1000).to_csv("NLP_3rdWeek_Sample.csv",index=False)

inData.loc[:,TEXT_COL]  = inData[['call_reason','call_subject','call_result']].apply(lambda x: ' '.join(map(lambda y: str(y),x)),axis=1)
inData.drop(columns=['call_reason','call_subject', 'call_result'],inplace=True)

inData.drop_duplicates(subset=['customer_id', 'contact_id',TEXT_COL],inplace=True)
print(inData.shape)

inData.loc[:,'TEXT_COL_CLEANED']  = inData[TEXT_COL].apply(clear_text)

inData.loc[:,'testOut'] = inData[TEXT_COL].apply(lambda x: str(x).find('test')!=-1)
print(inData.shape)
inData = inData[~pd.isnull(inData[TEXT_COL])]
print(inData.shape)

inData = inData[inData['testOut']==False]
inData.drop(columns='testOut',inplace=True)
print(inData.shape)


os.chdir(workDir+"\\week_3_task")

#Picking only the values in 2017
inData = inData[inData.creation_date<pd.Timestamp('2018-01-01 00:00:01')]
inData.to_csv("NLP_Master_Dataset.csv",index=False)

print(Sentiments_table.info())

for senti_name,senti_df in Sentiments_table.groupby('sentiment'):
    print(senti_name,senti_df.shape)
    vocabulary = np.array(list(set(senti_df.word.values)))
    cnt_vec = CountVectorizer(ngram_range=(1, 3),vocabulary=vocabulary)
    ASD = cnt_vec.transform(inData['TEXT_COL_CLEANED'])
    inData.loc[:,'WORD_'+str(senti_name)] = np.array(ASD.sum(axis=1)).ravel()

senti_COLS = list(Sentiments_table['sentiment'].unique())

inData['TOTAL_SCORED_WORDS'] = inData['WORD_POSITIVE'] + inData['WORD_NEGATIVE'] + inData['WORD_NEUTRAL']
inData['OVERALL_SENTIMENT'] = (inData['WORD_POSITIVE'] - inData['WORD_NEGATIVE'])/inData['TOTAL_SCORED_WORDS']

inData.fillna(value={'OVERALL_SENTIMENT':0},inplace=True)
#custData = inData.groupby('customer_id')['OVERALL_SENTIMENT'].agg(np.mean)
#custData = pd.DataFrame(custData).reset_index()
#custData.to_csv("CustomerLevelSenti.csv",index=False)

inData.to_csv("NLP_Sentiment_Output.csv",index=False)
#ASD = cnt_vec.transform(inData['call_result_cleaned']).sum(axis=0)
#ASD = np.array(ASD)[0]

#FREQ = pd.DataFrame([(i,j) for i,j in enumerate(ASD)],columns=['id','freq'])
#
#VOCAB = pd.DataFrame.from_dict(cnt_vec.vocabulary_,orient='index').reset_index()
#VOCAB.columns = ['word','id']
#
#VOCAB = VOCAB.merge(FREQ,how='left',on='id')
#
#Sentiments = pd.read_csv("senti_word.csv")
#VOCAB = VOCAB.merge(Sentiments,how='left',on='word')
#VOCAB.drop(columns='id',inplace=True)
#VOCAB.to_csv("VOCAB_SENTIWORD_LIST.csv",index=False)

#nltk.word_tokenize(inData.loc[0:1000,'call_result'])
