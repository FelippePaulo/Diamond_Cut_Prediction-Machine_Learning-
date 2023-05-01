# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:30:40 2023

@author: Felippe
"""

import pandas as pd
import matplotlib.pyplot as plt

################## CLassificação com KNN #####################

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []


# testando diferentes valores de K: de 1 a 50
k_settings = range(1, 51)
for k in k_settings:
    # construir o modelo
    classificador = KNeighborsClassifier(n_neighbors = k, metric='minkowski', p=2)
        
    # Treinamento
    classificador.fit(previsores_treinamento, classe_treinamento)

    # Acuracia na base de treinamento
    training_accuracy.append(classificador.score(previsores_treinamento, classe_treinamento))
    
    # Acuracia na base de teste
    test_accuracy.append(classificador.score(previsores_teste, classe_teste))



#  plotagem dos dados
plt.plot(k_settings,training_accuracy, label="acuracia de treinamento" )
plt.plot(k_settings,test_accuracy, label="acuracia de teste" )
plt.ylabel("Acuracia")
plt.xlabel("valor de K")
plt.legend()






























