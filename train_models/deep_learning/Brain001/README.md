### Description
Un modèle simple qui prend en entrée (un tableau de nombres) et génère en sortie (un autre tableau de nombres). Il tente de comprendre la formule mathématique utilisée en arrière-plan pour prédire les sorties. Le modèle s'entraîne plusieurs fois pour fournir une prédiction précise après avoir été correctement entraîné.

### Avantage 
- Le modèle peut déterminer automatiquement quelle formule mathématique utiliser pour prédire les sorties attendues.
- Il est applicable à une grande variété de formules mathématiques de base.

### Inconvénient 
Il peut prendre un nombre (temps) mais pas une liste de données pour produire une liste de tableaux de températures, ce qui représente une perte de temps.

### Programme

```python
  
  
# 15 janvier 2025  

#************************************* Imports ******************************#  
  
import csv  
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
  
model.fit(x=entree, y=sortie, epochs=5475)  
  
#:::::::::::::::::::::::::::::::::::* Train *:::::::::::::::::::::::::::::::::#  
  
while True:  
    x = int(input('Nombre :'))  
    prediction = model.predict(np.array([x]))  
    print(f'prediction :{prediction} ')


```