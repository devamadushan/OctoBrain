# 22 janvier 2025
# crée le cerveau du model
from bdb import Breakpoint

# ************************************* Imports ******************************#

from keras import Model , layers
from trainer import train_brain



# ******************************** Imports Terminal **************************#

#   pip install tenserflow

# ___________________________________* Brain *_________________________________#


def create_brain():


    input1 = layers.Input(shape=(1,), name="Input_1")

    x1 = layers.Dense(64, activation="relu")(input1)  ## in
    x = layers.Dense(64, activation="relu")(x1)  # hyden
    output = layers.Dense(1, activation=None, name="output")(x)  # out

    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer="adamW", loss="mean_squared_error", metrics=["mae"])
    model.summary()

    return model


brain = create_brain()
data_brain = train_brain(brain)

