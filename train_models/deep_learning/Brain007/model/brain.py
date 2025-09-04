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


    date = layers.Input(shape=(1,), name="Date")
    time = layers.Input(shape=(1,), name="Time")
    van_uar_c = layers.Input(shape=(1,), name="van_uar_c")
    van_uar_f = layers.Input(shape=(1,), name="van_uar_f")
    pression_ex = layers.Input(shape=(1,), name="pression_ex")
    temp_consigne = layers.Input(shape=(1,), name="temp_consigne")
    temp_aller_f = layers.Input(shape=(1,), name="temp_aller_f")

    ## in layer

    x1 = layers.Dense(64, activation="relu")(date)
    x2 = layers.Dense(64, activation="relu")(time)
    x3 = layers.Dense(64, activation="relu")(van_uar_c)
    x4 = layers.Dense(64, activation="relu")(van_uar_f)
    x5 = layers.Dense(64, activation="relu")(pression_ex)
    x6 = layers.Dense(64, activation="relu")(temp_consigne)
    x7 = layers.Dense(64, activation="relu")(temp_aller_f)

    combined = layers.concatenate([x1, x2 , x3, x4, x5, x6, x7])

    x = layers.Dense(448, activation="relu")(combined)  # hyden

    temp_reprise = layers.Dense(1, activation=None, name="temp_reprise")(x)  # out

    model = Model(inputs=[date , time ,van_uar_c , van_uar_f , pression_ex , temp_consigne , temp_aller_f], outputs=temp_reprise)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()

    return model


brain = create_brain()
data_brain = train_brain(brain)

