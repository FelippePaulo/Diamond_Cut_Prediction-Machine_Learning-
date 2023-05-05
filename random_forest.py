# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:34:59 2020

@author: Felippe
"""

import pandas as pd

############## Classificação com Random Forest ##############

from sklearn.ensemble import RandomForestClassifier

classificador = RandomForestClassifier(n_estimators=11,
                                       max_features=5,
                                       criterion='gini',
                                       max_depth=3,
                                       random_state=0)

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# Testes
previsoes = classificador.predict(previsores_teste)


# Análise dos resultados (porcentagem de acertos e matriz de confusão)
from sklearn.metrics import confusion_matrix, accuracy_score

acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)


# Visualizando a importancia das características
import matplotlib.pyplot as plt
import numpy as np

n_features = previsores.columns.size
plt.barh(range(n_features), classificador.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Features")
