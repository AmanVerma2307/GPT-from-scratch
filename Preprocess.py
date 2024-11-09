###### Importing Libraries
import os
import re
import nltk
import torch
import torchtext
import numpy as np
import pandas as pd

###### Preprocessing
word_mapping = {"its":"it is",
                "thats": "that is",
                "aint": "is not",
                "arent": "are not",
                "cant":"cannot",
                "dont":"do not",
                "didnt":"did not",
                "couldve": "could have",
                "youre": "you are",
                "couldnt": "could not",
                 "doesnt": "does not",
                 "hasnt": "has not", "havent": "have not",
                  'youve':"you have", 'goin':"going",
                  "youll":'you will'
                  }

def clean(txt):

    """
    Function for cleaning text

    INPUTS:-
    1) txt: Input sentence
    2) label_flag: Boolean value for 

    OUTPUTS:-
    1) txt: Cleaned sentence
    """

    def word_contraction(txt):
        txt = txt.split()
        for i in range(len(txt)):
            word = txt[i]
            if word in word_mapping:
                txt[i] = word_mapping[word]
        return " ".join(txt)
    
    def remove_stopwords(text):
        text = text.split()
        stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in stopword]
        return " ".join(text)
    
    def remove_punct(text):
        text = "".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text))
        text = re.sub('[0-9]+', '', text)
        return text
    
    def remove_fullstop(text):
        txt = []
        for item in text.split():
            if(item != '.'):
                txt.append(item)
        return " ".join(txt)
    
    def lemmatizer(text):
        text = text.split()
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text]
        return " ".join(text)

    txt = txt.lower() # Lower case
    #txt = remove_punct(txt)  # Remove punctuations
    #txt = remove_stopwords(txt) # Stop word removal
    txt = re.sub(r'\(.*\)','',txt) # Remove (words)
    txt = re.sub(r'[^a-zA-Z0-9 ]','',txt) # Remove punctuations
    txt = re.sub('[0-9]+', '', txt)
    #txt = re.sub(r'\.',' . ',txt)
    #txt = remove_fullstop(txt)
    #txt = txt.replace("'s",'') # Apostaphe removal
    #txt = remove_punct(txt) # Puntuation removal
    txt = lemmatizer(txt) # Lemmatization
    txt = word_contraction(txt) # Word contraction
    return txt

def preprocess(csv_path):

    """
    Function to preprocess input .csv file into clean sentences and summary

    INPUTS:-
    1) csv_path: Path to the target .csv file  

    OUTPUTS:-
    1) X: (N,) dimensional list of cleaned sentences
    2) y: Coressponding clean summary of dimension (N,)
    """
    X = []
    df = pd.read_csv(csv_path,index_col=False)
    
    for idx, item in enumerate(df['Text'].to_list()):
        X.append(clean(item))
    return X

###### Testing
#X = preprocess('./Reviews_q2_main.csv')
#print(len(X))
#X,y = preprocess('./train.csv')
#print(X[10])
#print(100*'=')
#print(X[-1],y[-1])
#print(y)
#print(len(X),len(y))
#for i in range(len(X)):
#    if(i < 10):
#        print(len(X[i].split()))
#        print(X[i])
#        print('------------')
#    else:
#        break
#    print(y[i])
#    print('================================================')