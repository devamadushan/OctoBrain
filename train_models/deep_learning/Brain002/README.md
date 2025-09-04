### Description
Dans le but de récupérer un fichier `csv` contenant deux colonnes (Temps, Température) et d'entraîner un modèle à prédire la température en fonction du temps, cette fois avec des données limitée .
### Avantage 
- Le modèle peut déterminer automatiquement quelle formule mathématique utiliser pour prédire les sorties attendues.
- Il est applicable à une grande variété de formules mathématiques de base.

### Inconvénient 
Toutes les formules mathématiques, comme les fonctions périodiques, ne peuvent pas être directement comprises par l'optimiseur Adam.

### Programme

```python
  
# 15 janvier 2025  
# Un test pour récupérer un fichier CSV (IN) partiellement rempli et prédire les valeurs manquantes  
  
#************************************* Imports ******************************#  
  
import pandas as pd  
from keras import *  
import numpy as np  
  
#******************************** Imports Terminal **************************#  
#   pip install tensorflow  
#************************************* Imports ******************************#  
  
  
  
#----------------------------* lire le fichier IN (CSV) *---------------------#  
  
data = pd.read_csv('data/data_in/brain002.csv')  
entree = data['Temps'].values  
sortie = data['Temperature'].values  
  
#___________________________________* Model *_________________________________#  
  
model = Sequential()  
model.add(layers.Dense(3,input_shape=[1]))  #input  
model.add(layers.Dense(64)) #hyden  
model.add(layers.Dense(1)) #out  
  
#-------------------------------------* Compile *-----------------------------#  
  
model.compile(loss='mean_squared_error', optimizer='adam')  
  
model.fit(x=entree, y=sortie, epochs=5475)  
  
#:::::::::::::::::::::::::::::::::::* Test *:::::::::::::::::::::::::::::::::#  
  
while True:  
    x = int(input('Nombre :'))  
    prediction = model.predict(np.array([x]))  
    print(f'prediction :{prediction} ')
    
```