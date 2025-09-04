# 22 janvier 2025
# entrainer le cerveau avec les donnée qui ont éte generer


# ************************************* Imports ******************************#

import pandas as pd
import datetime
import numpy as np

# ******************************** Imports Terminal **************************#

#  pip install ....

# ----------------------------* lire le fichier IN (CSV) *---------------------#
def load_datas_train():
    data_train = pd.read_csv('../data/data_train/data_train.csv')
    entry_train = data_train['Temps'].values
    out_train = data_train['Temperature'].values
    return entry_train , out_train


#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

path = "../data/data_brain/brain.hdf5"

#||||||||||||||||||||||||||||||||||*  Nombre de epochs  *||||||||||||||||||||||#

epochs = 5000

# -------------------------------------* Compile *-----------------------------#
def train_brain(brain):
    x_train , y_train  = load_datas_train()
    start_time = datetime.datetime.now()
    brain.fit(x_train, y_train, epochs=epochs, batch_size=64)
    end_time = datetime.datetime.now()

    print(f'départ : {start_time}')
    print(f'fin : {end_time}')
    print(f'la durée : {start_time - end_time}')


    brain.save(path)
    return path


