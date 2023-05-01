# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:34:30 2019

@author: marco
"""
#####################################################################
# Não dividir em base de treinamento e base de testes

# Classificador
from sklearn.neural_network import MLPClassifier

# Este método procura distribuir melhor o número de instâncias de cada classe em cada fold
# Na verdade, este método apenas auxilia a divisão dos dados, a repetição da aplicação é feita na mão
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 3)
acuracias = []
matrizes = []
metricas = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):    
    classificador = MLPClassifier(verbose = True,
                              max_iter=1000,
                              tol = 0.000001,
                              solver = 'sgd',
                              hidden_layer_sizes=[10],
                              activation='relu',
                              random_state =1)
    classificador.fit(previsores[indice_treinamento], classe.iloc[indice_treinamento,0]) 
    previsoes = classificador.predict(previsores[indice_teste])
    acuracia = accuracy_score(classe.iloc[indice_teste,0], previsoes)
    
    metricas.append(precision_recall_fscore_support(classe.iloc[indice_teste,0], previsoes))
    matrizes.append(confusion_matrix(classe.iloc[indice_teste,0], previsoes))
    acuracias.append(acuracia)

######################## Resultado final ########################
# Matriz de confusão média
matriz_media = np.mean(matrizes, axis = 0)
matriz_desvio_padrao = np.std(matrizes, axis = 0)
# Métricas médias
acuracias = np.asarray(acuracias)
acuracia_final_media = acuracias.mean()
acuracia_final_desvio_padrao = acuracias.std()
metricas_medias = np.mean(metricas, axis = 0) 
metricas_desvio_padrao = np.std(metricas, axis = 0)
#Obs: linhas: precisao, recall, f1score // colunas: classe