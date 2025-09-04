### Description
Exactement la même chose que le modèle 003, mais cette fois avec 27 000 valeurs différentes de temps. Il récupère un fichier `CSV` en entrée et génère un fichier `CSV` contenant les prédictions.
### Avantage 
...

### Inconvénient 
Il prend beaucoup plus de temps de calcul, par exemple : environ 5 minutes pour traiter 27 000 valeurs.
### Programme
`create_data_train.py`
```python

# 15/01/2025  
# crée un fichier csv avec un tres grand nombre de donnée  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
# ******************************** Imports Terminal **************************#  
#   pip install ...  
  
  
# ----------------------------* crée un fichier (CSV) *---------------------#  
temps = list(range(1,501))  
param = 3  
temperature =[]  
for i in range(1,501):  
    temperature.append(10 * np.sin(2 * np.pi * param *i / 100)) 
  
df = pd.DataFrame({"Temps" : temps, "Temperature" : temperature} )  
df.to_csv("Brain004.csv" ,index=False)
```

`create_data_in.py`
```python

# 16 janvier 2025  
# Crée un fichier "data_in" avec une colonne "Temps" remplie de manière aléatoire  
# et une colonne "Temperature" laissée vide  
  
# ************************************* Imports ******************************#  
  
import pandas as pd  
import numpy as np  
  
# ******************************** Imports Terminal **************************#  
#   pip install ...  
  
  
# ----------------------------* crée un fichier IN (CSV) *---------------------#  
  
temps = []  
temperature = []  
  
for i in range(1, 27000):  
    temps.append(np.random.randint(1, 9951200))  
  
  
data_in_csv = pd.DataFrame({"Temps": temps , "Temperature": np.nan})  
data_in_csv.to_csv("Brain004.csv" ,index=False)

```


`brain004.py`
```python


```