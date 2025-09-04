# date du derniere modification
# objectif


# ************************************* Imports ******************************#
from keras import  layers, models,regularizers
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import numpy as np
from keras import saving
import datetime
import json
# ******************************** Imports Terminal **************************#
#   pip install ...

# **************************** Lire le fishier params Json  *******************#

with open('params.json', "r", encoding="utf-8") as f:
    data = json.load(f)

data_input = data['path_data_train']['data_input']
data_output = data['path_data_train']['data_output']
data_brain = data['path_brain']


# -------------------------------------* paths *-----------------------------#

#path_inputs_data = 'data/inputs/inputs.csv'
#path_outputs_data = 'data/output/output.csv'


#************************************  seed ***********************************#

seed = 42
np.random.seed(seed)

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

#path_brain = "../data/brain/brain.hdf5"

# ----------------------------* Main programme *-----------------------------#


def pickups_datas(path_inputs_data, path_outputs_data):
    x = pd.read_csv(path_inputs_data)
    y = pd.read_csv(path_outputs_data)

    print("nulllllllll",x.isnull().sum())
    print("nullll",y.isnull().sum())
    y.isnull().sum()
    print("xxxxxxxxxxxxxxxxxxx",x)
    print("yyyyyyyyyyyyyyyyyyy",y)

    print("Testing Input shape\t: {}".format(x.shape))
    print("Training Output shape\t: {}".format(y.shape))
  #  print("Training Data\t: {}",x)
    # Standardiser uniquement X
    #std_clf = StandardScaler()
    #x_train_new = std_clf.fit_transform(x_train)
    #x_test_new = std_clf.transform(x_test)

    # Vérifier que les features sont bien alignées
    #x_train_new = pd.DataFrame(x_train_new, columns=x_train.columns)
    #x_test_new = pd.DataFrame(x_test_new, columns=x_train.columns)

    # One-Hot Encoding pour y_train si nécessaire
    #encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    #print("x train new last\t: {}", x_train_new)
    return x, y


def train(brain ,path_inputs_data,path_outputs_data,path_brain , epochs):
    x_train ,  y_train = pickups_datas(path_inputs_data , path_outputs_data)

    start_time = datetime.datetime.now()
    brain.fit(x_train, y_train, epochs=epochs, batch_size=7)

    end_time = datetime.datetime.now()
    duree = end_time - start_time
    print(f'départ : {start_time}')
    print(f'fin : {end_time}')
    print(f'la durée : {duree}')

    saving.save_model(brain, path_brain)
    return duree , format(x_train.shape)


