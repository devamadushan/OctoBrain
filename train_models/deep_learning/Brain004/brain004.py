# 16 janvier 2025
# Un test pour récupérer un fichier CSV (IN) partiellement rempli et prédire les valeurs manquantes avec cette fois plus de donnée et une formule de mathématique plus complex

# ************************************* Imports ******************************#

import pandas as pd
from keras import *
import numpy as np
import datetime
from data.data_train import create_data_train
from data.data_in import create_data_in


# ******************************** Imports Terminal **************************#
#   pip install tenserflow


# ----------------------------* lire le fichier IN (CSV) *---------------------#

data_train = pd.read_csv('data/data_train/Brain004.csv')

entree_train = data_train['Temps'].values
sortie_train = data_train['Temperature'].values

# ___________________________________* Model *_________________________________#

model = Sequential()
model.add(layers.Dense(16, input_shape=[1]))  # input
model.add(layers.Dense(32))  # hyden
model.add(layers.Dense(1))  # out

# -------------------------------------* Compile *-----------------------------#
avant = datetime.datetime.now()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x=entree_train, y=sortie_train, epochs=5500)
apres = datetime.datetime.now()
print(f'départ : {avant}')
print(f'fin : {apres}')

print(f'la durée : {apres - avant}')
#:::::::::::::::::::::::::::::::::::* Test *:::::::::::::::::::::::::::::::::#

while True:
    data_in_present = int(input("si vous avez charger le fichier csv dans data_in entrez 1 : "))
    if data_in_present == 1:
        data_in = pd.read_csv('data/data_in/Brain004.csv')

        entree_in = data_in['Temps'].values
       # print(entree_in)

        prediction = model.predict(entree_in)
        data_out = prediction.flatten()
        #print(data_out)
        df = pd.DataFrame({"Temps": entree_in,"Temperature":data_out})
        df.to_csv('data/data_out/Brain004.csv', index=False)
        data_in_present = 0
    else:
        print("saisie incorrect")

