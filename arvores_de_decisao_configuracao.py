# -*- coding: utf-8 -*-
"""
Created on Mon may 1 17:39:55 2023

@author: Felippe
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

training_accuracy = []
test_accuracy = []

# tentando diferentes valores de K: de 1 a 30
h = range(1,31)
for i in h:
    # Construir o modelo
    classificador = DecisionTreeClassifier(max_depth=i,
                                           criterion='entropy',
                                           random_state=0)
    # Treinamento
    classificador.fit(previsores_treinamento, classe_treinamento)
    # resultado na base de treinamento
    training_accuracy.append(classificador.score(previsores_treinamento, classe_treinamento))
    # resultado na base de teste
    test_accuracy.append(classificador.score(previsores_teste, classe_teste))

plt.plot(h, training_accuracy, label="training accuracy")
plt.plot(h, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend()
