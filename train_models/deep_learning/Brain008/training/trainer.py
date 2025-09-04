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
# ******************************** Imports Terminal **************************#
#   pip install ...

# -------------------------------------* paths *-----------------------------#

#path_inputs_data = 'data/inputs/inputs.csv'
#path_outputs_data = 'data/output/output.csv'


#************************************  seed ***********************************#
seed = 42
np.random.seed(seed)

#************************************ Epoche ***********************************#

epochs = 3000
#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

#path_brain = "../data/brain/brain.hdf5"

# ----------------------------* Main programme *-----------------------------#


def pickups_datas(path_inputs_data, path_outputs_data):
    x = pd.read_csv(path_inputs_data)
    y = pd.read_csv(path_outputs_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed, shuffle=True
    )

    print("Training Input shape\t: {}".format(x_train.shape))
    print("Testing Input shape\t: {}".format(x_test.shape))
    print("Training Output shape\t: {}".format(y_train.shape))
    print("Testing Output shape\t: {}".format(y_test.shape))
    print("Training Data\t: {}",x)
    # Standardiser uniquement X
    std_clf = StandardScaler()
    #x_train_new = std_clf.fit_transform(x_train)
    #x_test_new = std_clf.transform(x_test)

    # Vérifier que les features sont bien alignées
    #x_train_new = pd.DataFrame(x_train_new, columns=x_train.columns)
    #x_test_new = pd.DataFrame(x_test_new, columns=x_train.columns)

    # One-Hot Encoding pour y_train si nécessaire
    #encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    #print("x train new last\t: {}", x_train_new)
    return x, y


def train(brain ,path_inputs_data,path_outputs_data,path_brain):
    x_train ,  y_train = pickups_datas(path_inputs_data , path_outputs_data)

    start_time = datetime.datetime.now()
    brain.fit(x_train, y_train, epochs=epochs, batch_size=7)

    end_time = datetime.datetime.now()

    print(f'départ : {start_time}')
    print(f'fin : {end_time}')
    print(f'la durée : {end_time - start_time}')

    saving.save_model(brain, path_brain)
    return path_brain


