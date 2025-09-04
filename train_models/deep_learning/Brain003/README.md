### Description
Dans le but de récupérer un fichier `csv` contenant deux colonnes (Temps, Température) et d'entraîner un modèle à prédire la température en fonction du temps, cette fois avec plus de données .
### Avantage 
....

### Inconvénient 
Toutes les formules mathématiques, comme les fonctions périodiques, ne peuvent pas être directement comprises par l'optimiseur Adam.

### Programme

`create_data_train.py`
```python

# 15/01/2025  
# crée un fichier csv avec un tres grand nombre de donnée  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
  
# ******************************** Imports Terminal **************************#  
#   pip install ...  
  
  
# ----------------------------* crée un fichier (CSV) *---------------------#  
temps = list(range(1,501))  
temperature = [temp * 2 for temp in temps]  
  
df = pd.DataFrame({"Temps" : temps, "Temperature" : temperature} )  
df.to_csv("Brain003.csv" ,index=False)  
  
```

`brain003.py`
```python

# 16 janvier 2025  
# Un test pour récupérer un fichier CSV (IN) partiellement rempli et prédire les valeurs manquantes avec cette fois plus de donnée  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
from keras import *  
import numpy as np  

# ******************************** Imports Terminal **************************#  
#   pip install tenserflow  
  
  
# ----------------------------* lire le fichier IN (CSV) *---------------------#  
  
data = pd.read_csv('data/data_train/Brain003.csv')  
  
entree =  data['Temps'].values  
sortie = data['Temperature'].values  
  
# ___________________________________* Model *_________________________________#  
  
model = Sequential()  
model.add(layers.Dense(16, input_shape=[1])) #input  
model.add(layers.Dense(32)) #hyden  
model.add(layers.Dense(1)) #out  
  

# -------------------------------------* Compile *-----------------------------#  
model.compile(loss='mean_squared_error',optimizer='adam')  
model.fit(x=entree, y=sortie, epochs=5475)  
  
#:::::::::::::::::::::::::::::::::::* Test *:::::::::::::::::::::::::::::::::#  
  
while True:  
    x = int(input('Nombre :'))  
    prediction = model.predict(np.array([x]))  
    print(f'prediction :{prediction} ')
```