# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:18:04 2019

@author: Felippe
"""

import pandas as pd

################## Classificação com Árvores de Decisão ################## 

# Geração da árvore
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(max_depth=2,
                                       criterion='entropy', 
                                       random_state=0)

# Treinamento 
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)

# Resultados na base de treinamento, para verificar overfitting
previsoes_treinamento = classificador.predict(previsores_treinamento)
acuracia_treinamento = accuracy_score(classe_treinamento, previsoes_treinamento)
matriz_treinamento = confusion_matrix(classe_treinamento, previsoes_treinamento)

# Exportando a árvore para fazer figura
from sklearn.tree import export_graphviz
export_graphviz(classificador,out_file="tree.dot",class_names=["bad","good","3","4"],
                feature_names=cols_previsores, impurity=False, filled=True)

# Visualizando a árvore
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# Visualizando a importância das características
import matplotlib.pyplot as plt
import numpy as np
n_features = previsores.columns.size
plt.barh(range(n_features), classificador.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")



