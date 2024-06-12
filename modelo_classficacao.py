# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:51:33 2024

@author: leoja
"""

#%% Bibliotecas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


#%% Leitura do dataset
mnist = fetch_openml('mnist_784',version=1)

# Mostra as chaves do dict
mnist.keys()

# Seleciona os dados de interesse
x,y = mnist['data'].values, mnist['target'].values
y = y.astype(float)

# Pixels dos número
x

# Número correspondente
y

# Reorganiza o dado e mostra o número
n = 2
plt.imshow(x[n].reshape(28, 28), cmap='binary')
print(y[n])
plt.show()

#%% Classificador Binário
from sklearn.model_selection import train_test_split

# Separa o conjunto de dados em dados de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Define o número 5 como True e o resto como False
y_train_5 = (y_train==5)
pd.Series(y_train_5).value_counts()

from sklearn.linear_model import SGDClassifier

# Treina o modelo de classificação
sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_5)

# Exemplo de predição
n = 15
plt.imshow(x_train[n].reshape(28, 28), cmap='binary')
print('Classe real: ',y_train_5[n])
print('Classe predita pelo modelo: ', sgd_clf.predict([x_train[n]]))

# Mede a acurácia do modelo
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy')

# Modelo que mostra que todos os valores não é 5, ou seja, Falso
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(sef, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring='accuracy')

# Matriz de confusão
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, sgd_clf.predict(x_train))

# Métricas de classificação
from sklearn.metrics import precision_score, recall_score

# Predição dos dados de treino
y_train_pred = sgd_clf.predict(x_train)

# Cálculo da precisão e recall
print('Precision:', precision_score(y_train_5, y_train_pred))
print('Recall:', recall_score(y_train_5, y_train_pred))

# Métrica que mostra a precisão, recall e f1-score
from sklearn.metrics import classification_report
print(classification_report(y_train_5, y_train_pred))

# Método para prever qualquer número
sgd_clf.fit(x_train, y_train)

# Mostra os valores
n = 2
digit = x_train[n]
plt.imshow(digit.reshape(28,28))
sgd_clf.predict([digit])

# Cross-validation predict
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)

# Matriz de confusão e métricas
conf_mx = confusion_matrix(y_train,y_train_pred)
print(conf_mx)
print(classification_report(y_train, y_train_pred))

# Heatmap da matriz de confusão
fig, ax = plt.subplots(figsize=(25,8))
sns.heatmap(conf_mx, annot=True, fmt='.0f')

# Matriz normalizada
row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
fig, ax = plt.subplots(figsize=(30,10))
sns.heatmap(norm_conf_mx, annot=True)

# Classificação Multilabel
from sklearn.neighbors import KNeighborsClassifier

# Números maiores iguais a 7
y_train_large = (y_train >= 7)

# Número impares
y_train_odd = (y_train % 2 == 1)

# Junta os dois conjuntos
y_multilabel = np.c_[y_train_large, y_train_odd]

# Treina o modelo
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)

# Mostra os valores
n = 4
digit = x_train[n]
plt.imshow(digit.reshape(28,28))
knn_clf.predict([digit])











