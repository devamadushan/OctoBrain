# 17 janvier 2025
# le cerveau du model
from bdb import Breakpoint

# ************************************* Imports ******************************#

from keras import *
from trainer import train_brain


# ******************************** Imports Terminal **************************#

#   ppip install tenserflow

# ___________________________________* Brain *_________________________________#


def create_brain():

    brain = Sequential()
    brain.add(layers.Dense(16, input_shape=[1]))  # input
    brain.add(layers.Dense(32))  # hyden
    brain.add(layers.Dense(1))  # out

    return brain


brain = create_brain()
data_brain = train_brain(brain)
