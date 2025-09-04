# 15/01/2025
# crée un fichier csv avec un tres grand nombre de donnée

# ************************************* Imports ******************************#

import pandas as pd
# ******************************** Imports Terminal **************************#

#   pip install ...

# ----------------------------* crée un fichier (CSV) *---------------------#
temps = list(range(1,27))
param = 3
temperature =[]
for i in range(1,27):
    temperature.append(param * i)

df = pd.DataFrame({"Temps" : temps, "Temperature" : temperature} )
df.to_csv("data_train.csv" ,index=False)
