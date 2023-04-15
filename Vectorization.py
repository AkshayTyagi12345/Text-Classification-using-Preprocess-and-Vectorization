import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
def MakeCorpus(list_of_strings):
    allWords=[]
    for each in list_of_strings:
        words=nltk.word_tokenize(each)
        for every in words:
            allWords.append(every)
    corpus=list(set(allWords))
    return corpus

def PresenceAbsenceVectorization(list_of_string):
    corpus=MakeCorpus(list_of_string)
    list_of_vectors=[]
    for each in list_of_string:
        words=nltk.word_tokenize(each)
        vector=[]
        for every in corpus:
            if every in words:
                vector.append(1)
            else:
                vector.append(0)
        list_of_vectors.append(vector)
    return list_of_vectors

def CountVectorization(list_of_string):
    corpus=MakeCorpus(list_of_string)
    list_of_vectors=[]
    for each in list_of_string:
        words=nltk.word_tokenize(each)
        vector=[]
        for every in corpus:
            if every in words:
                vector.append(words.count(every))
            else:
                vector.append(0)
        list_of_vectors.append(vector)
    return list_of_vectors

from sklearn.feature_extraction.text import TfidfVectorizer
def TFIDFVectorization(list_of_string):
    vectorizer=TfidfVectorizer()
    tf_idfMatrix=vectorizer.fit_transform(list_of_string).toarray()
    return list(tf_idfMatrix)
