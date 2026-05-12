# anas-project – Détection des avis clients (Analyse de sentiment)

Ce projet propose une solution complète et professionnelle pour la **détection automatique du sentiment** à partir d’avis clients écrits en français ou en anglais. L’objectif est de déterminer si un avis est *positif*, *négatif* ou *neutre* en s’appuyant sur plusieurs approches d’apprentissage automatique, puis de mettre à disposition une **interface utilisateur moderne** pour tester le modèle en temps réel ou en lot.

## Aperçu rapide

- **Entrée** : avis texte (`review`) en CSV ou saisi manuellement.
- **Sortie** : sentiment prédit (`positive` | `negative` | `neutral`) et score de confiance.
- **UI** : app Streamlit pour tester en ligne ou en batch et télécharger les résultats annotés.
- **Modèles fournis** : Logistic Regression et Linear SVM pré‑entraînés sur l’échantillon fourni, avec sélection automatique du meilleur via `best_model.txt`.
- **Usage cible** : support client, modération, suivi e‑réputation.

## Choix technologiques

| Aspect | Choix | Justification |
|---|---|---|
| **Langage** | Python 3 | Large écosystème pour le NLP, librairies matures, support de Streamlit. |
| **Modèles classiques** | TF‑IDF + Logistic Regression / Linear SVM | Modèles rapides à entraîner et performants pour des jeux de données structurés【557036607249085†L344-L350】. |
| **Modèle avancé** | CamemBERT (ou tout autre modèle Transformer via Hugging Face) | Les transformers pré‑entraînés comme CamemBERT surpassent nettement les approches classiques sur le français【557036607249085†L344-L351】. |
| **Interface graphique** | Streamlit | Permet de créer rapidement une interface web moderne et réactive sans se soucier de HTML/CSS. |
| **Gestion de pipeline** | Scikit‑learn pour les modèles classiques, `transformers`/`datasets` pour le modèle avancé | APIs unifiées pour l’entraînement, l’évaluation et la sérialisation. |

## Structure du projet

```text
answer/
├── app/
│   └── streamlit_app.py    # Interface graphique Streamlit
├── data/
│   └── sample_reviews.csv   # Exemple de dataset (avis + sentiment)
├── models/                  # Dossier où sont enregistrés les modèles entraînés
├── src/                     # Modules Python du projet
│   ├── preprocess.py        # Fonctions de nettoyage et de chargement des données
│   ├── train.py             # Script d’entraînement et de sélection du meilleur modèle
│   ├── evaluate.py          # Script d’évaluation des modèles sur le jeu de test
│   └── predict.py           # Chargement des modèles et prédictions sur de nouveaux textes
├── README.md                # Ce document
└── requirements.txt         # Liste des dépendances Python
```

### Description des répertoires et fichiers

* **app/** : contient l’application Streamlit. Lancer `streamlit run app/streamlit_app.py` pour accéder à l’interface.
* **data/** : contient un fichier d’exemple `sample_reviews.csv` avec deux colonnes : `review` (texte de l’avis) et `sentiment` (étiquette parmi `positive`, `negative`, `neutral`). Remplacez ce fichier par votre propre CSV pour entraîner le modèle sur vos données.
* **models/** : répertoire où sont sauvegardés les modèles (`logreg_model.joblib`, `svm_model.joblib`, éventuellement `bert_model/`) et un fichier `best_model.txt` indiquant le meilleur modèle selon la validation.
* **src/** : code source du pipeline :
  * `preprocess.py` : fonctions de nettoyage du texte et de chargement du dataset.
  * `train.py` : sépare les données en train/validation/test, entraîne les modèles classiques (LogReg et SVM), lance éventuellement un fine‑tuning CamemBERT, compare les performances (basé sur le F1‑score de validation) et sauvegarde les modèles et les métriques.
  * `evaluate.py` : charge le dataset et les modèles sauvegardés, calcule les métriques sur le jeu de test et génère des matrices de confusion.
  * `predict.py` : utilitaire pour charger le meilleur modèle et prédire le sentiment d’un texte ou d’un fichier CSV.

## Installation

### Prérequis

- Python 3.10+ (testé avec 3.11)
- (Optionnel) GPU pour le fine‑tuning Transformer
- Windows : exécuter les commandes dans PowerShell

1. **Cloner le projet** (si nécessaire) ou télécharger le dossier `answer/`.
2. Créez un environnement virtuel (recommandé) :

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

   > **Remarque :** l’entraînement d’un modèle Transformer nécessite `torch`, `transformers` et `datasets`. Ces paquets sont inclus dans `requirements.txt` mais l’entraînement peut être long sans GPU.

## Entraînement des modèles

Le script `train.py` charge un fichier CSV contenant les colonnes `review` et `sentiment`, effectue un nettoyage léger, divise les données en **train/validation/test**, entraîne plusieurs modèles et sélectionne automatiquement le meilleur. Les modèles sont sauvegardés dans `models/`.

### Commande d’entraînement (modèles classiques uniquement)

```bash
python -m src.train --data data/sample_reviews.csv --out_dir models
```

### Commande avec fine‑tuning CamemBERT (optionnel)

```bash
python -m src.train \
    --data data/mon_dataset.csv \
    --out_dir models \
    --train_bert \
    --bert_model camembert-base
```

Le fine‑tuning peut durer plusieurs minutes à plusieurs heures en fonction de la taille du dataset et des ressources disponibles.

## Évaluation

Après l’entraînement, évaluez les performances sur le jeu de test via :

```bash
python -m src.evaluate --data data/mon_dataset.csv --model_dir models --out_dir reports
```

Ce script génère un fichier `reports/evaluation_report.txt` contenant l’accuracy, la précision, le rappel et le F1‑score pour les modèles LogReg et SVM, ainsi que des matrices de confusion enregistrées au format PNG.

## Prédictions

### Prédire un avis à partir de la ligne de commande

```bash
python -m src.predict --model_dir models --text "Je recommande fortement ce produit !"
```

### Prédire en masse sur un fichier CSV

```bash
python -m src.predict --model_dir models --input_csv data/nouveaux_avis.csv --output_csv resultats.csv
```

Le fichier `resultats.csv` contiendra deux colonnes supplémentaires : `predicted_sentiment` et `confidence`.

## Interface Streamlit

Lancez l’application web avec :

```bash
streamlit run app/streamlit_app.py
```

Fonctionnalités :

* Saisie manuelle d’un avis avec retour immédiat du sentiment et du score de confiance.
* Choix d’un exemple pré‑défini dans une liste (positif, négatif, neutre).
* Téléversement d’un fichier CSV d’avis pour prédiction en lot ; affichage d’un tableau des résultats et d’un graphique montrant la distribution des sentiments.
* Téléchargement du CSV annoté.
* Visualisation des mots les plus influents pour le modèle Logistic Regression via la barre latérale.

## Format du dataset

Votre fichier d’entraînement doit être au format CSV et contenir au minimum :

| Colonne   | Description                                  |
|-----------|----------------------------------------------|
| `review`  | Texte de l’avis client                      |
| `sentiment` | Étiquette parmi `positive`, `negative`, `neutral` |

Des valeurs manquantes seront ignorées. Les colonnes supplémentaires sont conservées mais ne sont pas utilisées pour l’entraînement.

## Suggestions d’améliorations futures

* **Augmentation des données** : intégrer des jeux de données publics plus volumineux (par ex. le jeu de données Allociné pour le français, contenant 100 000 avis positifs et 100 000 négatifs【557036607249085†L285-L289】) et des ensembles multilingues contenant la classe neutre.
* **Hyper‑paramétrisation** : réaliser une recherche de grille pour optimiser les paramètres des modèles classiques (C, ngram_range, etc.) ou utiliser des techniques d’optimisation bayésienne.
* **Entraînement multi‑langue** : fine‑tuner un modèle multilingue (e.g. `xlm‑roberta-base`) capable de gérer plusieurs langues et de distinguer trois sentiments.
* **Interprétabilité avancée** : intégrer des méthodes d’explicabilité (LIME, SHAP) pour visualiser l’impact des mots sur les prédictions des modèles Transformers.
* **Déploiement** : conteneurisation via Docker et mise en ligne sur un PaaS (Heroku, Streamlit Community Cloud, etc.).

---

## Démarrage rapide sous Windows (exécution de l’app)

```powershell
cd answer
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> Les dépendances `numpy==1.26.4` et `pandas==2.3.3` sont déjà verrouillées dans `requirements.txt` pour éviter les blocages de DLL sur Windows.

## Modèles inclus par défaut

- `models/logreg_model.joblib` : régression logistique entraînée sur `data/sample_reviews.csv`.
- `models/svm_model.joblib` : SVM linéaire entraîné sur le même jeu.
- `models/best_model.txt` : contient le nom du modèle sélectionné après validation. Le code de prédiction charge ce modèle par défaut.

## Flux de traitement

```text
CSV (review, sentiment)
   ↓ nettoyage + split (train/val/test)
   ↓ vectorisation TF‑IDF
   ↓ entraînement LogReg + SVM (+ CamemBERT optionnel)
   ↓ sélection du meilleur F1 (validation)
   ↓ sauvegarde modèles + métriques
   ↓ prédiction (CLI ou Streamlit)
```

## Dépannage courant

- **Erreur “DLL load failed” sur NumPy** : déjà contournée via `numpy==1.26.4`. Si le problème réapparaît après réinstallation globale de Python, réexécutez `pip install numpy==1.26.4` dans le venv.
- **Port 8501 déjà utilisé** : lancer `streamlit run app/streamlit_app.py --server.port 8502`.
- **Problèmes d’encodage CSV** : ajouter `--encoding utf-8` côté chargement ou convertir le fichier en UTF‑8.

### Références

Les résultats comparant CamemBERT à d’autres modèles sur un grand corpus d’avis sont tirés du dépôt de Théophile Blard sur GitHub【557036607249085†L285-L351】. Cette source montre qu’un modèle CamemBERT atteint une accuracy de ~97 % sur le jeu de données Allociné, dépassant largement les modèles classiques【557036607249085†L344-L350】.
>>>>>>> 0799c23 (Add customer sentiment analysis app)
