# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:29:36 2023

@author: Felippe
"""

import pandas as pd


################## CLassificação com KNN #####################

from sklearn.neighbors import KNeighborsClassifier

#  n_neighbors é o valor de K
# Minkowski com p=2 é a distancia euclidiana
classificador = KNeighborsClassifier(n_neighbors = 60, metric='minkowski', p=2)

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)


# Teste
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e Matriz de Confusão)
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)
