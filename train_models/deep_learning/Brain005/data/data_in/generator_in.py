# 17 janvier 2025
# generer une donnée vide pour que l cerveaux le trouver


# ************************************* Imports ******************************#

import pandas as pd
import numpy as np

# ******************************** Imports Terminal **************************#
#   pip install ...

# ----------------------------* crée un fichier IN (CSV) *---------------------#

temps = []
temperature = []

for i in range(1, 27):
    temps.append(np.random.randint(1, 995))


data_in_csv = pd.DataFrame({"Temps": temps , "Temperature": np.nan})
data_in_csv.to_csv("data_in.csv" ,index=False)

