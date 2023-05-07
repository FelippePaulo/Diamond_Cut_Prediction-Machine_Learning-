# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:25:31 2023

@author: Felippe
"""


import pandas as pd

################## #Pré-processamento dos dados ################## 

#Leitura dos dados
base = pd.read_csv('output.csv')
resumo = base.describe()

# Procurando valores inconsistentes
#base.loc[base['Age'] < 0]

# Apagar a coluna
# base.drop('Age', 1, inplace=True)

# Apagar apenas os registros com problema
# base.drop(base[base.Age < 0].index, inplace=True)

# Preencher os valores com a média
#age_media = base['Age'][base.Age > 0].mean()
#base.loc[base.Age < 0, 'Age'] = age_media

# traducaoClarity = ["'IF'","'VVS1'","'VVS2'","'VS1'","'VS2'","'SI1'","'SI2'","'I1'"]
# #base.loc[base.clarity == "'SI2'", 'clarity'] = 6
# cont = 0
# for i in traducaoClarity:
#     base.loc[base.clarity == i, 'clarity'] = cont 
#     cont = cont + 1
# del i

#  "cut": {
#        "Fair": 0,
#        "Good": 1,
#        "Very Good": 2,
#        "Premium": 3,
#        "Ideal": 4
#    },
# traducaoCut = ["'Fair'","'Good'","'Very Good'","'Premium'","'Ideal'"]
# cont = 0
# for i in traducaoCut:
#     base.loc[base.cut == i, 'cut'] = int(cont) 
#     cont += 1

# del i,cont,traducaoCut,traducaoClarity


# Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()

# Preencher os valores com o mais frequente
#saving_maioria = base['Saving accounts'][pd.notnull(base['Saving accounts'])].describe().top
#base.loc[pd.isnull(base['Saving accounts']), 'Saving accounts'] = saving_maioria

# Separando dados em previsores e classes
cols_previsores = ['carat','color','clarity','depth','table','price','\'x\'','\'y\'','\'z\'']
cols_classe = ['cut']
previsores = base[cols_previsores]
classe = base[cols_classe]

# Transforma as variáveis categóricas em valores numéricos     
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
# previsores.loc[:, 'color'] = labelencoder_previsores.fit_transform(previsores.loc[:, 'color'])
#previsores.loc[:, 'cut'] = labelencoder_previsores.fit_transform(previsores.loc[:, 'cut'])
previsores.loc[:, 'clarity'] = labelencoder_previsores.fit_transform(previsores.loc[:, 'clarity'])

#traducao para clarity
#"clarity": {
#        "IF": 0,
#        "VVS1": 1,
#        "VVS2": 2,
#        "VS1": 3,
#        "VS2": 4,
#        "SI1": 5,
#        "SI2": 6,
#        "I1": 7

#from sklearn.preprocessing import LabelEncoder
labelencoder_cut = LabelEncoder()
categorias = ["'Fair'","'Good'","'Very Good'","'Premium'","'Ideal'"]
labelencoder_cut.fit(categorias)
classe.loc[:, 'cut'] = labelencoder_cut.transform(classe.loc[:, 'cut'])
del categorias

bins = [-1, 1, 2, 3, 4.1]
labels = ["Fair", "Good","Very good", "Ideal"]
classe["cut"] = pd.cut(classe["cut"], bins=bins, labels=labels)
classe.loc[:,"cut"] = labelencoder_cut.fit_transform(classe.loc[:,"cut"])

# labelencoder_cut = LabelEncoder()
# categorias = ["Fair","Good","Very Good","Ideal"]
# labelencoder_cut.fit(categorias)
# classe.loc[:, 'cut'] = labelencoder_cut.transform(classe.loc[:, 'cut'])
# del categorias


# Criando variáveis dummy para categoricas nominais
from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()

#  Variável color
variaveis_dummy = labelbinarizer.fit_transform(previsores['color'])
novas_variaveis_dummy = labelbinarizer.classes_
df_variaveis_dummy = pd.DataFrame(variaveis_dummy, columns=labelbinarizer.classes_)
# trocando variavel antiga pelas novas variaveis com previsores
previsores=previsores.join(df_variaveis_dummy)
previsores=previsores.drop('color', axis=1)



cols_previsores = previsores.columns



# Padronização dos dados
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)

# Separando em base de testes e treinamento (usando 25% para teste)
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# #####################################################################
