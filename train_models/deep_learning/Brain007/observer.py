# 22 janvier 2025
# simuler des donnée


# ************************************* Imports ******************************#

import pandas as pd
import tensorflow as  tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ******************************** Imports Terminal **************************#
#   pip install ...

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

path_brain = 'data/data_brain/brain.hdf5'

#++++++++++++++++++++++++++++++++++* path data in *+++++++++++++++++++++++++#

path_in = 'data/data_in/data_in.csv'

#++++++++++++++++++++++++++++++++++* path data out *+++++++++++++++++++++++++#

path_out = 'data/data_out/data_out.csv'

#++++++++++++++++++++++++++++++++++* path data real *+++++++++++++++++++++++++#

path_true = 'data/data_real/data_real.csv'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* parametre in  *~~~~~~~~~~~~~~~~~~~~~~~~#


param1 = "Date"
param2 = "Time"
param3 = "VANNE_UTA_CHAUD [I]"
param4 = "VANNE_UTA_FROID [I]"
param5 = "PRESSION_EXTERIEUR [R]"
param6 = "TEMPERATURE_CONSIGNE [R]"
param7 = "TEMPERATURE_ALLER_FROID [R]"
result1 = "TEMPERATURE_REPRISE [R]"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* charger les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#

def load_data_truth():
    data_truth = pd.read_csv(path_true , sep=';')
   # data_1 = data_truth[f'{param1}'].values
   # data_2 = data_truth[f'{param2}'].values
   # data_3 = data_truth[f'{param3}'].values
   # data_4 = data_truth[f'{param4}'].values
   # data_5 = data_truth[f'{param5}'].values
   # data_6 = data_truth[f'{param6}'].values
   # data_7 = data_truth[f'{param7}'].values
    out_1 = data_truth[f'{result1}'].values
    return out_1



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* enregistrer les datas *~~~~~~~~~~~~~~~~~~~~~~~~~#



def find_truth(entry_pred , out_pred):
    y1 = load_data_truth()
    return y1

#:::::::::::::::::::::::::::::::::::*  play *::::::::::::::::::::::::::::::::::#
