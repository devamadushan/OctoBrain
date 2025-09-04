# 17 janvier 2025
# entrainer le cerveau avec les donnée qui ont éte generer


# ************************************* Imports ******************************#

import pandas as pd
import datetime
import tensorflow as tf

# ******************************** Imports Terminal **************************#

#  pip install ....

# ----------------------------* lire le fichier IN (CSV) *---------------------#
data_train = pd.read_csv('../data/data_train/data_train.csv')
entry_train = data_train['Temps'].values
out_train = data_train['Temperature'].values


#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#
path = "../data/data_brain/brain.hdf5"
# -------------------------------------* Compile *-----------------------------#
def train_brain(brain):

    start_time = datetime.datetime.now()
    brain.compile(loss='mean_squared_error', optimizer='adam')
    brain.fit(x=entry_train, y=out_train, epochs=5000)
    end_time = datetime.datetime.now()
    print(f'départ : {start_time}')
    print(f'fin : {end_time}')
    print(f'la durée : {start_time - end_time}')
    brain.save(path)
    return path


