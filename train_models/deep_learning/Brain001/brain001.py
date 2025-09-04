

# 15 janvier 2025


#************************************* Imports ******************************#

from keras import *
import numpy as np

#******************************** Imports Terminal **************************#
#   pip install tensorflow


#___________________________________* Model *_________________________________#

model = Sequential()
model.add(layers.Dense(3,input_shape=[1]))  #input
model.add(layers.Dense(64)) #hyden
model.add(layers.Dense(1)) #out

entree = np.array([1,2,3,4,5])
sortie = np.array([2,4,6,8,10])

#-------------------------------------* Compile *-----------------------------#

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=entree , y=sortie, epochs=5475)

#:::::::::::::::::::::::::::::::::::* Train *:::::::::::::::::::::::::::::::::#

while True:
    x = int(input('Temps :'))
    prediction = model.predict(np.array([x]))
    print(f'Temperature  :{prediction} ')