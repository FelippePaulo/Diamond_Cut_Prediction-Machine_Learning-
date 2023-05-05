# -*- coding: utf-8 -*-
"""
Created on Apr 28 13:00:41 2023

@author: Felippe 
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


training_accuracy = []
test_accuracy = []

# tentando diferentes valores de K: de 1 a 50
k = range(1,11)
for i in k:
    # Construir o modelo
    classificador = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    # Treinamento
    classificador.fit(previsores_treinamento, classe_treinamento)
    # resultado na base de treinamento
    training_accuracy.append(classificador.score(previsores_treinamento, classe_treinamento))
    # resultado na base de teste
    test_accuracy.append(classificador.score(previsores_teste, classe_teste))
    print("----------------------------------------------- : " + str(i))
plt.plot(k, training_accuracy, label="training accuracy")
plt.plot(k, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend()