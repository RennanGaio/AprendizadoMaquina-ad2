'''aluno: Rennan de Lucena Gaio'''

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_tnc

from sklearn.naive_bayes import GaussianNB

#import dataset to python
data = pd.read_csv('dataset3.tsv', sep='\t')
data = data.loc[data['Sentiment'].isin([0,4])]

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

#transforma o dataset de target em 0 e 1 ao invez de 0 e 4, sendo 0 negativo e 1 positivo
for idx, e in enumerate(y):
    if e == 4:
        y[idx]=1

#transform the text dataset to a bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
#print(X.toarray())

gnb = GaussianNB()
modelo = gnb.fit(X, Y)
