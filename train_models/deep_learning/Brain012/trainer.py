# date du derniere modification
# objectif


# ************************************* Imports ******************************#
import pandas as pd
import numpy as np
from keras import saving
import datetime
import json


# ******************************** Imports Terminal **************************#
#   pip install ...

# **************************** Lire le fishier params Json  *******************#

# with open('params.json', "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# data_input = data['path_data_train']['data_input']
# data_output = data['path_data_train']['data_output']
# data_brain = data['path_brain']


# -------------------------------------* paths *-----------------------------#

#path_inputs_data = 'data/inputs/inputs.csv'
#path_outputs_data = 'data/output/output.csv'


#************************************  seed ***********************************#

seed = 42
np.random.seed(seed)

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#

#path_brain = "../data/brain/brain.hdf5"

# ----------------------------* Main programme *-----------------------------#


class Trainer:

    def __init__(self , path_brain ,path_inputs_data , path_output_data , brain , epochs):
        self.path_brain = path_brain
        self.input_data = path_inputs_data
        self.output_data = path_output_data
        self.brain = brain
        self.epochs = epochs

    def train_modele(self):
        x_train, y_train = self.pickups_datas()

        start_time = datetime.datetime.now()
        self.brain.fit(x_train, y_train, epochs=self.epochs, batch_size=7)

        end_time = datetime.datetime.now()
        duree = end_time - start_time
        print(f'départ : {start_time}')
        print(f'fin : {end_time}')
        print(f'la durée : {duree}')

        saving.save_model(self.brain, self.path_brain)
        return duree, format(x_train.shape)

    def pickups_datas(self):
        x = pd.read_csv(self.input_data)
        y = pd.read_csv(self.output_data)

        print("Testing Input shape\t: {}".format(x.shape))
        print("Training Output shape\t: {}".format(y.shape))

        return x, y
