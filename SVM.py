# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:45:52 2020

@author: marco
"""

import pandas as pd

######### Classificacao com SVM ##############

# Treinamento
from sklearn.svm import SVC
classificador = SVC(kernel = 'rbf', C = 0.9, gamma = 'scale', random_state = 1)
classificador.fit(previsores_treinamento, classe_treinamento)

# Testes
previsoes = classificador.predict(previsores_teste)


# Análise dos resultados (porcentagem de acertos e matriz de confusão)
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)