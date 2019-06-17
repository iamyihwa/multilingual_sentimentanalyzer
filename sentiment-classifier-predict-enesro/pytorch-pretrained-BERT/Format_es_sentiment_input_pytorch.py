import pandas as pd
#from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from pandas import DataFrame

#le = LabelEncoder()

df = pd.read_csv('data/es_3_sentiment/train.tsv', sep = '\t')

#creating train and dev dataframes according to Bert 
df_bert_train = pd.DataFrame({'row_id': range(df.shape[0]), 
			'label': df['polarity'], #le.fit_transform(df['polarity']), 
			'alpha': ['a']*df.shape[0], 
			'text': df['text']})

df_valid = pd.read_csv('data/es_3_sentiment/valid.tsv', sep = '\t')
df_bert_valid = pd.DataFrame({'row_id': range(df_valid.shape[0]),
                        'label': df_valid['polarity'], #le.fit_transform(df_valid['polarity']),
                        'alpha': ['a']*df_valid.shape[0],
                        'text': df_valid['text']})

df_test = pd.read_csv('data/es_3_sentiment/test.tsv', sep = '\t')
df_bert_test = pd.DataFrame({'row_id': range(df_test.shape[0]),
                        'text': df_test['text']})

df_bert_train.to_csv('data/es_3_sentiment_bert/train.tsv',sep='\t', index=False, header=False)
df_bert_valid.to_csv('data/es_3_sentiment_bert/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/es_3_sentiment_bert/test.tsv',sep='\t', index=False, header=False)
