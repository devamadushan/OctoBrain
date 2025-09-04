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