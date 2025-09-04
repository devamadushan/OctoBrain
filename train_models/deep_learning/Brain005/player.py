# 17 janvier 2025
# similer des donnée


# ************************************* Imports ******************************#

import pandas as pd
import tensorflow as  tf
from tensorflow import keras


# ******************************** Imports Terminal **************************#
#   pip install ...
path = 'data/data_brain/brain.hdf5'
#:::::::::::::::::::::::::::::::::::* play *:::::::::::::::::::::::::::::::::#

while True:
    data_in_present = int(input("si vous avez charger le fichier csv dans data_in entrez 1 : "))
    if data_in_present == 1:
        data_in = pd.read_csv('data/data_in/data_in.csv')
        octopus_brain = tf.keras.models.load_model(path)
        entree_in = data_in['Temps'].values
        print(entree_in)

        prediction = octopus_brain.predict(entree_in)
        data_out = prediction.flatten()
        #print(data_out)
        df = pd.DataFrame({"Temps": entree_in,"Temperature":data_out})
        df.to_csv('data/data_out/data_out.csv', index=False)
        data_in_present = 0
    else:
        print("saisie incorrect")
