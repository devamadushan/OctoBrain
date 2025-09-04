# 15/01/2025
# crée un fichier csv avec un tres grand nombre de donnée

# ************************************* Imports ******************************#

import pandas as pd

# ******************************** Imports Terminal **************************#
#   pip install ...


# ----------------------------* crée un fichier (CSV) *---------------------#
temps = list(range(1, 501))
temperature = [temp * 2 for temp in temps]

df = pd.DataFrame({"Temps": temps, "Temperature": temperature})
df.to_csv("Brain003.csv", index=False)
