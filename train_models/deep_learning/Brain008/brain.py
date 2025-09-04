# 29 janvier 2025
# le cerveau du model


# ************************************* Imports ******************************#

from keras import  layers, models,regularizers , losses , optimizers
import numpy as np
from keras.src.saving import register_keras_serializable
import tensorflow as tf
from training.trainer import train
from keras import saving

# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# -------------------------------------* paths *-----------------------------#

path_inputs_data = 'training/data/inputs/inputs.csv'
path_outputs_data = 'training/data/output/output.csv'
path_brain = "data/brain/brain3.hdf5"
# ___________________________________* Brain *_________________________________#


def get_path_brain():
    return path_brain


def create_brain():
    model = models.Sequential()

    # Inputs
    model.add(layers.Dense(32, input_dim=5, activation='relu',
                           kernel_initializer='he_normal' , kernel_regularizer=regularizers.l2(0.001)
                           ))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))

    # Hidden Layer
    model.add(layers.Dense(64, activation='relu',
                           kernel_initializer='he_normal' , kernel_regularizer=regularizers.l2(0.001)
                           ))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))

    # Output
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=optimizers.AdamW(
        learning_rate=0.001, weight_decay=1e-5
    ), loss='mean_squared_error', metrics=['mae'])
    model.summary()
    return model


brain = create_brain()
train(brain , path_inputs_data, path_outputs_data , path_brain)