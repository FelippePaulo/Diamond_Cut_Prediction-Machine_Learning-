# -*- coding: utf-8 -*-
"""
Created on may 5  17:15:52 2023

@author: Felippe
"""

import pandas as pd

############## Classificacao com Redes Neurais ###################

#Treinamento
from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.000001,
                              solver = 'sgd',
                              hidden_layer_sizes = [10],
                              activation = 'relu',
                              random_state = 1)
classificador.fit(previsores_treinamento, classe_treinamento)

# Testes
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e matriz de confusão)
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)


# Plotagem do mapa de calor da primeira camada escondida
import matplotlib.pyplot as plt
plt.imshow(classificador.coefs_[0], cmap='viridis')
plt.yticks(range(len(cols_previsores)), cols_previsores)
plt.xlabel("Colunas na matriz de pesos")
plt.ylabel("Variáveis previsoras")
plt.colorbar()


# Avaliação de métricas
import numpy as np

from sklearn.metrics import classification_report
valores_classe = np.unique(base[classe.columns].values)
metricas_resumo = classification_report(classe_teste, previsoes, target_names=valores_classe)