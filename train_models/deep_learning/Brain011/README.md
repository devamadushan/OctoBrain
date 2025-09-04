# Brain010

Ce dossier contient les scripts et les données nécessaires pour le projet Octobrain, qui vise à créer un jumeau numérique d'une cellule climatique.

## Arborescence détaillée

- **Scripts principaux** :
  - `brain.py` : Script central qui définit et entraîne le modèle. Il utilise Keras (TensorFlow) pour créer un réseau de neurones qui apprend à prédire le comportement de la cellule climatique.
  - `formater.py` : Script pour formater les données avant l'entraînement.
  - `filter.py` : Script pour filtrer les données.
  - `trainer.py` : Script pour entraîner le modèle.
  - `tester.py` : Script pour tester le modèle.
  - `player.py` : Script pour simuler le comportement du modèle.
  - `smartFrame.py` : Script pour gérer les données de manière intelligente.
  - `echantillonneur.py` : Script pour échantillonner les données.

- **Données** :
  - `data_train.csv` : Fichier contenant les données d'entraînement.
  - `data_test.csv` : Fichier contenant les données de test.
  - `params.json` : Fichier de configuration pour les paramètres du modèle.

- **Dossiers** :
  - `data_raw/` : Dossier contenant les données brutes.
  - `train/` : Dossier pour les données d'entraînement.
  - `test/` : Dossier pour les données de test.
  - `play/` : Dossier pour les simulations.

## Structure

Le projet est structuré de manière à séparer les différentes étapes du processus :
1. **Préparation des données** : Les scripts `formater.py` et `filter.py` sont utilisés pour préparer les données.
2. **Entraînement** : Le script `trainer.py` est utilisé pour entraîner le modèle.
3. **Test** : Le script `tester.py` est utilisé pour évaluer la performance du modèle.
4. **Simulation** : Le script `player.py` est utilisé pour simuler le comportement du modèle.

## Comment ça marche

1. **Installation des dépendances** : Assure-toi d'avoir les dépendances nécessaires installées (Keras, TensorFlow, Pandas). Tu peux les installer avec pip :
   ```bash
   pip install keras tensorflow pandas
   ```
2. **Configuration des paramètres** : Modifie le fichier `params.json` pour ajuster les paramètres du modèle selon tes besoins.
3. **Exécution du modèle** : Exécute `brain.py` pour entraîner et tester le modèle. Le script va lire les données, entraîner le modèle, et sauvegarder les résultats.

## Dépendances

- **Keras** : Une bibliothèque de haut niveau pour construire et entraîner des modèles de deep learning.
- **TensorFlow** : Une bibliothèque open source pour le machine learning, utilisée comme backend par Keras.
- **Pandas** : Une bibliothèque pour la manipulation et l'analyse de données, utilisée pour gérer les données tabulaires. 