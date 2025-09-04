### Description
Ce modèle peut sauvegarder l'IA dans le répertoire `data_brain` une fois que l'entraînement, basé sur les données placées dans le répertoire `data_train`, est terminé. Ensuite, un autre programme permet de faire des prédictions en fonction de ce sur quoi il a été entraîné, ainsi que de prédire des valeurs à partir d'un fichier `CSV` fourni dans le répertoire `data_in`
### Avantage 
Enrichissement de l'arborescence et séparation des programmes (`trainer` et `player`), qui sont complètement indépendants. Dans les modèles précédents, ils étaient tous interconnectés, ce qui signifiait que toute modification, même mineure, dans le `player` nécessitait de recommencer l'entraînement avant de pouvoir exécuter le programme. Cela représentait une perte de temps.

### Architecture
```
data : contient toutes les données  
  data_in : le fichier CSV avec des données partiellement remplies  
    data_in.csv  
    generator_in.py : permet de générer le fichier CSV  
  data_out : le fichier CSV généré par le modèle avec les prédictions  
  data_train : le fichier CSV utilisé pour l'entraînement  
    data_train.csv  
    generator_train.py : permet de générer un fichier CSV  
  data_brain : contient le modèle entraîné  
    brain.kras  

model : contient  
  brain.py : crée le modèle  
  trainer.py : permet d'entraîner le modèle  

player.py : permet de faire des prédictions  
README.md : contient les informations sur le projet  

```
### Inconvénient 
Il prend beaucoup plus de temps de calcul, par exemple : environ 5 minutes pour traiter 27 000 valeurs.
### Programme
`generator_train.py`
```python

# 15/01/2025  
# crée un fichier csv avec un tres grand nombre de donnée  
  
# ************************************* Imports ******************************#  
import pandas as pd  
# ******************************** Imports Terminal **************************#  
  
#  pip install ...  
  
# ----------------------------* crée un fichier (CSV) *---------------------#  


temps = list(range(1,27))  
param = 3  
temperature =[]  
for i in range(1,27):  
    temperature.append(param * i)  
  
df = pd.DataFrame({"Temps" : temps, "Temperature" : temperature} )  
df.to_csv("data_train.csv" ,index=False)

```

`brain.py`
```python

# 17 janvier 2025  
# le cerveau du model  
from bdb import Breakpoint  
  
# ************************************* Imports ******************************#  
  
from keras import *  
from trainer import train_brain  
  
# ******************************** Imports Terminal **************************#  
  
#   pip install tenserflow  
  
# _______________________________* creation Brain *__________________________#  
  
  
def create_brain():  
  
    brain = Sequential()  
    brain.add(layers.Dense(16, input_shape=[1]))  # input  
    brain.add(layers.Dense(32))  # hyden  
    brain.add(layers.Dense(1))  # out  
  
    return brain  
  
  
brain = create_brain()  
data_brain = train_brain(brain) #path ou elle se trouver le cerveau
```


`trainer.py`
```python

# 17 janvier 2025  
# entrainer le cerveau avec les donnée qui ont éte generer  
  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
import datetime  
# ******************************** Imports Terminal **************************#  
  
#  pip install ....  
  
# ----------------------------* lire le fichier IN (CSV) *---------------------#  
data_train = pd.read_csv('../data/data_train/data_train.csv')  
entry_train = data_train['Temps'].values  
out_train = data_train['Temperature'].values  
  
  
#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#  
path = "../data/data_brain/brain.hdf5"  

# ------------------------------------* Compile *------------------------------#  
def train_brain(brain):  
  
    start_time = datetime.datetime.now()  
    brain.compile(loss='mean_squared_error', optimizer='adam')  
    brain.fit(x=entry_train, y=out_train, epochs=5000)  
    end_time = datetime.datetime.now()  
    print(f'départ : {start_time}')  
    print(f'fin : {end_time}')  
    print(f'la durée : {start_time - end_time}')  
    brain.save(path)  
    return path
    
```


`player.py`
```python
# 17 janvier 2025  
# similer des donnée  
  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
import tensorflow as  tf  
 
  
# ******************************** Imports Terminal **************************#  
#   pip install ...  

#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#
path = 'data/data_brain/brain.hdf5'  
#:::::::::::::::::::::::::::::::::::* play *:::::::::::::::::::::::::::::::::::#  
  
while True:  
    data_in_present = int(input("si vous avez charger le fichier csv dans data_in entrez 1 : "))  
    if data_in_present == 1:  
        data_in = pd.read_csv('data/data_in/data_in.csv')  
        octopus_brain = tf.keras.models.load_model(path)  
        entree_in = data_in['Temps'].values  
        print(entree_in)  
  
        prediction = octopus_brain.predict(entree_in)  
        data_out = prediction.flatten()  
        #print(data_out)  
        df = pd.DataFrame({"Temps": entree_in,"Temperature":data_out})  
        df.to_csv('data/data_out/data_out.csv', index=False)  
        data_in_present = 0  
    else:  
        print("saisie incorrect")
```



`generator_in.py`
```python
# 17 janvier 2025  
# entrainer le cerveau avec les donnée qui ont éte generer  
  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
import datetime  
# ******************************** Imports Terminal **************************#  
  
#  pip install ....  
  
# ----------------------------* lire le fichier IN (CSV) *---------------------#  
data_train = pd.read_csv('../data/data_train/data_train.csv')  

entry_train = data_train['Temps'].values  
out_train = data_train['Temperature'].values  
  
  
#++++++++++++++++++++++++++++++++++* path data brain *+++++++++++++++++++++++++#  
path = "../data/data_brain/brain.hdf5"  
# -------------------------------------* Compile *-----------------------------#  
def train_brain(brain):  
  
    start_time = datetime.datetime.now()  
    brain.compile(loss='mean_squared_error', optimizer='adam')  
    brain.fit(x=entry_train, y=out_train, epochs=5000)  
    end_time = datetime.datetime.now()  
    print(f'départ : {start_time}')  
    print(f'fin : {end_time}')  
    print(f'la durée : {start_time - end_time}')  
    brain.save(path)  
    return path
```
