
## Description

## Rôle
Le projet **Octopus Brain** est un outil destiné à créer des modèles simples et à les conserver pour pouvoir les réutiliser.

--- 

## Outil
**version : 013**
### Modules
- **SmartFrame** : traite les données
       - **Échantillonneur** : échantillonne les données  
    - **Formater** : corrige et formate les données
    - **Filter** : sépare les données en deux parties (in / out)
- **Trainer** : entraîne le modèle
- **Tester** : évalue le modèle
- **Player** : interroge les modèles existants
- **Cleaner** : nettoie les données résiduelles de l’outil et sauvegarde les métadonnées
## Paramétrés du modèle
Dans ce projet, nous avons utilisé un modèle Sequential en deep learning.  
L’architecture est composée de :
- Couche d’entrée dense, normalisée avec _BatchNormalization_ et régularisée avec _Dropout (20%)_
- Couche cachée dense de 64 neurones, suivie de _BatchNormalization_ et _Dropout (20%)_
- Couche de sortie dense adaptée à la tâche de régression
- Optimiseur : _AdamW_
- Fonction de perte : adaptée à la régression (ex. _MSE_)
- Métriques de suivi : _MAPE_ et _MSE_

### Métadonnées
- Noms des paramètres (entrée, sortie)
- Données d’entraînement (in / out)
- Résultats des indicateurs d’évaluation (MAE, MSE, R²)
- Temps d’entraînement
- Fonction de perte (_loss_) utilisée
### Métriques d’évaluation
- MAE : (Mean Absolute Error)
	- Faible -> GOOD , Fort -> BAD
	Mesure la moyenne des écarts absolus entre les valeurs prédites et les valeurs réelles.
- FORMULE :
	$MAE=1n∑i=1n∣yi−y^i∣\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|MAE=n1​i=1∑n​∣yi​−y^​i​∣$

- MSE : (Mean Squared Error) 
	-  Faible -> GOOD , Fort -> BAD
	La MSE mesure la moyenne des carrés des écarts entre les valeurs réelles et prédites. Elle pénalise fortement les grosses erreurs.
- FORMULE:
	$MSE=n1​i=1∑n​(yi​−y^​i​)2$
	
- R²: (R-squared) 
	- 1 -> parfait ; 0 -> moyen , <0 -> faible
	Le R² mesure la proportion de la variance des données expliquée par le modèle. C’est un score global de qualité.
- FORMULE : 
	$R2=1−∑(yi​−yˉ​)2∑(yi​−y^​i​)2​$

### Modéles
- **brain_1** : un jumeau numérique permettant de simuler la température de reprise en fonction des conditions des cellules (CEREEP–Ecotron Île-de-France).
- **brain_2** : un jumeau numérique pour simuler la température de reprise d’une cellule climatique (CEREEP–Ecotron Île-de-France).


### Arborescence
```bash
├── brain.py          # Module cerveau de l'outil, contrôle tous les autres modules
├── brains            # Dossier où sont stockés les modèles entraînés
│   # Contient les modèles sauvegardés après entraînement
├── cleaner.py        # Module qui nettoie les données et sauvegarde les métadonnées
├── data_raw
│   ├── infer
│   │   └── # Données pour les futures simulations
│   ├── test
│   │   └── # Dépôt des données de test sans échantillonnage
│   ├── to_client
│   │   └── # Données simulées par le player
│   ├── to_sample
│   │   └── # Endroit où déposer toutes les données brutes
│   └── train
│       └── # Données brutes pour l’entraînement sans échantillonnage
├── echantillonneur.py # Module qui échantillonne les données
├── filter.py          # Module qui sépare les données (in/out)
├── formater.py        # Module qui corrige et nettoie les données
├── params.json        # Fichier de configuration des paramètres
├── play               # Données du player
│   └── data
│       ├── formated_data
│       │   └── # Données nettoyées
│       ├── input
│       │   └── # Données d'entrée
│       └── output
│           └── # Données de sortie
├── player.py          # Module pour interroger un modèle existant
├── README.md          # Documentation
├── smartFrame.py      # Module qui regroupe échantillonneur, formateur et filtre
├── test               # Données liées au module testeur
│   └── data
│       ├── formated_data
│       │   └── # Données nettoyées
│       ├── input_test
│       │   └── # Données d'entrée pour le test
│       ├── output_test
│       │   └── # Données de sortie du modèle
│       └── simulated
│           └── # Données simulées par le modèle
├── tester.py          # Module qui permet de tester un modèle
├── tracer.py          # Module qui trace un schéma
├── train              # Données liées au module entraîneur
│   └── data
│       ├── formated_data
│       │   └── # Données nettoyées
│       ├── input_train
│       │   └── # Données d'entrée pour l’entraînement
│       └── output_train
│           └── # Données de sortie pour l’entraînement
├── trainer.py         # Module qui permet d’entraîner un modèle et de l’enregistrer
```


![[logo.webp]]





