# 22 janvier 2025
# generer une donnée vide pour que l cerveaux le trouver


# ************************************* Imports ******************************#

import pandas as pd
import numpy as np

# ******************************** Imports Terminal **************************#
#   pip install ...

# ----------------------------* crée un fichier IN (CSV) *---------------------#

temps = np.linspace(0, 2 * np.pi, 19000)
temperature = []




data_in_csv = pd.DataFrame({"Temps": temps , "Temperature": np.nan})
data_in_csv.to_csv("data_in.csv" ,index=False)

