# 16 janvier 2025
# Un test pour récupérer un fichier CSV (IN) partiellement rempli et prédire les valeurs manquantes avec cette fois plus de donnée

# ************************************* Imports ******************************#

import pandas as pd
from keras import *
import numpy as np
# ******************************** Imports Terminal **************************#
#   pip install tenserflow


# ----------------------------* lire le fichier IN (CSV) *---------------------#

data = pd.read_csv('data/data_train/Brain003.csv')

entree =  data['Temps'].values
sortie = data['Temperature'].values

# ___________________________________* Model *_________________________________#

model = Sequential()
model.add(layers.Dense(16, input_shape=[1])) #input
model.add(layers.Dense(32)) #hyden
model.add(layers.Dense(1)) #out




# -------------------------------------* Compile *-----------------------------#
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x=entree, y=sortie, epochs=15475)

#:::::::::::::::::::::::::::::::::::* Test *:::::::::::::::::::::::::::::::::#

while True:
    x = int(input('Nombre :'))
    prediction = model.predict(np.array([x]))
    print(f'prediction :{prediction} ')