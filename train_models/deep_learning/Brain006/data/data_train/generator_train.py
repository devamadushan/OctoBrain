# 22 janvier 2025
# generer des donnée  pour que le cerveaux s'entraine


# ************************************* Imports ******************************#

import pandas as pd
import numpy as np




# ******************************** Imports Terminal **************************#
#   pip install ...

#œœœœœœœœœœœœœœœœœœœœœœœœœœœœœœœ* fonction mathematique *œœœœœœœœœœœœœœœœœœœœœœœ#

def find_truth(valeur):
    return np.sin(valeur)

# ----------------------------* crée un fichier IN (CSV) *---------------------#

temps = np.linspace(0, 2 * np.pi, 1000)
temperature = []

for i in range(0 , len(temps)):
    val = temps[i]
    temperature.append(find_truth(val))


data_in_csv = pd.DataFrame({"Temps": temps , "Temperature": temperature})
data_in_csv.to_csv("data_train.csv" ,index=False)


