
import pandas as pd
import numpy as np
import nltk
import streamlit as st
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def clean_text(text_list):
    # recebe uma lista de strings
    twitter_punct = '!"$\'*+,-./?`{|}~%:'
    clean_text_list = []
    for text in text_list:
        # pontuações
        text_sempont = [char for char in text if char not in twitter_punct]
        text_sempont = ''.join(text_sempont)

        # remover links
        text_semlink = re.sub('https\S+', '', text_sempont)

        # stopwords
        text_process = [word for word in text_semlink.split()
                        if word.lower() not in stopwords.words('portuguese')]
        text_process = ' '.join(text_process)

        clean_text_list.append(text_process)

    # retorna uma lista de strings limpa
    return clean_text_list

def main():
    st.title('Predição de Sentimento em Tweet - Deploy')

    df = pd.read_csv('tweets_processados.csv')
    df = df.dropna()
    df = df.drop_duplicates(['Text'])

    tweets = df.Text
    classes = df.Classificacao

    estimator = st.selectbox('Selecione o estimador',
                             ('Naive Bayes', 'RandomForest', 'SVM'))

    if estimator  == 'Naive Bayes':
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=TweetTokenizer().tokenize)),
            ('tf-idf', TfidfTransformer()),
            ('Naive Bayes', MultinomialNB())
        ])

    elif estimator == 'RandomForest':
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=TweetTokenizer().tokenize)),
            ('tf-idf', TfidfTransformer()),
            ('Random Forest', RandomForestClassifier())
        ])

    else:
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=TweetTokenizer().tokenize)),
            ('tf-idf', TfidfTransformer()),
            ('SVM', SVC())
        ])


    pipeline.fit(tweets, classes)
    tweets_test = [st.text_input('Digite aqui o tweet!')]
    tweet_cleaned = clean_text(tweets_test)
    tweet_button = st.button('Pronto!')
    if tweet_button:
        predicao = pipeline.predict(tweet_cleaned)
        st.markdown(predicao[0])

main()