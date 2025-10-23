#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-23T16:06:46.110Z
"""

# # **Credit Card Fraud Detection Project** 💳⚠️
# 
# Ce projet **vise à détecter les transactions frauduleuses** sur des cartes de crédit en utilisant des techniques de **Data Science et Machine Learning**.  
# 
# Le dataset utilisé contient les colonnes suivantes :  
# 
# - **Time** ⏰ : Le temps écoulé (en secondes) depuis la première transaction dans le dataset.  
# - **V1, V2, ..., V28** 🔢 : Variables anonymisées issues d'une **transformation PCA** pour protéger la confidentialité des clients.  
# - **Amount** 💰 : Montant de la transaction.  
# - **Class** 🚨 : Indique si la transaction est **frauduleuse (1)** ou **non frauduleuse (0)**.  
# 
# ## **Objectifs du projet** 🎯
# 1. **Analyser le dataset** pour comprendre la distribution des transactions et des fraudes.  
# 2. **Prétraiter les données** (normalisation, gestion du déséquilibre de classes, etc.).  
# 3. **Entraîner des modèles de machine learning** pour détecter les fraudes.  
# 4. **Évaluer les performances** des modèles avec des métriques adaptées comme la **precision, recall et F1-score**.  
# 5. Fournir une **solution prédictive efficace** pour aider les institutions financières à **réduire les pertes liées on des fraudes**.
# précision 🎯**  learn, XGBoost 🏎️  
# 


# # **Imports et Préparation de l'Environnement** 🛠️
# premièrement on va oc **imporrte toutes les bibliothèques et modules nécessaires** pour le projet de détection de fraude sur les cartes de crédit. Il prépare l'environnement pour :
# 
# - **Manipulation et analyse des données** : `pandas`, `numpy`  
# - **Visualisation** : `matplotlib`, `seaborn`  
# - **Statistiques et transformations** : `scipy`, `stats`, `boxcox`  
# - **Prétraitement des données** : `sklearn.preprocessing`, `StandardScaler`  
# - **Modélisation et apprentissage automatique** :  
#   - Régression et classification : `LogisticRegression`, `Ridge`, `Lasso`  
#   - Arbres et ensembles : `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`  
#   - Support Vector Machines et KNN : `SVC`, `KNeighborsClassifier`  
#   - XGBoost : `XGBClassifier`, `plot_importance`  
# - **Validation et recherche d'hyperparamètres** : `train_test_split`, `KFold`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, `RepeatedKFold`  
# - **Gestion du déséquilibre des classes** : `RandomOverSampler`, `imblearn.over_sampling`  
# - **Autres outils utiles** : gestion du temps (`time`), suppression des warnings tion de fraudes.
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn import over_sampling
import sklearn
import time
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

import warnings 
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from datetime import datetime

mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment("CreditCard_Fraud_Detection")


# 
# D'abord on va **charger le dataset des transactions par carte de crédit** depuis le fichier `creditcard.csv` dans un **DataFrame pandas** nommé `credit_card_transactions_data`.  
# 
# Le DataFrame contiendra toutes les colonnes du dataset, y compris les **variables anonymisées (V1 à V28)**, le **montant de la transaction (Amount)** et la **classe indiquant la fraude (Class)**.  
# 
# Cela constitue la **base pour toutes les étapes suivantes** du projet : exploration, prétraitement, modélisation et évaluation des modèles.
# 


credit_card_transactions_data = pd.read_csv("creditcard.csv")

credit_card_transactions_data.head()

# Cette ligne affiche un **aperçu des informations du dataset** `credit_card_transactions_data`.  
# 
# Elle permet de connaître :  
# - Le **nombre total de transactions** : 284 807.  
# - Le **nombre et le type de colonnes** : 31 colonnes au total, dont 30 **float64** (Time, V1 à V28, Amount) et 1 **int64** (C).s ).  
# - La **présence de valeurs manquantes** : aucolonne ne contient de valeurs nulles


credit_card_transactions_data.info()

credit_card_transactions_data.isnull().sum()

# Cette ligne permet de **comprendre la distribution des classes** dans le dataset `credit_card_transactions_data`.  
# 
# - **0 (non frauduleuse)** : 284 315 transactions  
# - **1 (frauduleuse)** : 492 transactions  
# 
# On remarque que le dataset est **très déséquilibré**, avec une proportion très faible de fraudeaude.
# 


credit_card_transactions_data['Class'].value_counts()

# Cette commande calcule et **visualise la proportion de chaque classe** dans le dataset.  
# 
# - Elle montre que les transactions **non frauduleuses (Class 0)** représentent environ **99,83 %** des données.  
# - Les transactions **frauduleuses (Class 1)** représentent seulement **0,17 %** des données.  
# 
# La commande `.plot.pie()` crée un **diagramme circulaire** pour visualiser ce déséquilibre, ce qui aide à **comprendre l'importance de traiter le problème des classes déséquilibrées** avant de former un modèle.
# 


print((credit_card_transactions_data.groupby("Class")["Class"].count() / credit_card_transactions_data["Class"].count()) * 100)
((credit_card_transactions_data.groupby("Class")["Class"].count() / credit_card_transactions_data["Class"].count()) * 100).plot.pie()

# Ce bloc crée un **graphique pour visualiser le nombre de transactions par classe**.  
# 
# - `sns.countplot` affiche le **nombre de transactions non frauduleuses (Class 0)** et **frauduleuses (Class 1)**.  
# - Les axes sont étiquetés pour montrer le **nombre d'enregistrements par classe**.  
# - Le titre met en évidence que le graphique représente le **comptage des classes**.  
# 


plt.figure(figsize=(7,5))
sns.countplot(x='Class', data=credit_card_transactions_data)
plt.title("Class count" , fontsize=18)
plt.xlabel("Record count by class", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

# Ce bloc calcule et **visualise la matrice de corrélation** du dataset .  
# 
# - `credit_card_transactions_data.corr()` calcule la **corrélation entre toutes les colonnes numériques**.  
# - `sns.heatmap` crée une **carte de chaleur** pour représenter visuellement ces corrélations, avec les valeurs annotées pour plus de clarté.  
# - La palette `coolwarm` permet de distinguer facilement les **corrélations positives et négatives**.  
# 
# Cette visualisation aide à **identifier les relations entre les variables** et peut guider la **sélection des caractéristiques** ou la **détection de variables fortement corrélées** avant la modélisation.
# 


corr = credit_card_transactions_data.corr()
corr

plt.figure(figsize=(24,18))

sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()

# Ce bloc transforme la colonne **`Time`** du dataset en plusieurs composantes temporelles pour faciliter l'analyse :  
# 
# - `pd.to_timedelta` convertit la colonne `Time` (en secondes) en **objet timedelta**.  
# - De cette transformation, on extrait :  
#   - **`Time_Day`** : le nombre de jours écoulés depuis la première transaction.  
#   - **`Time_Hour`** : l'heure de la transaction dans la journée.  
#   - **`Time_Min`** : les minutes de la transaction.  
# - Ensuite, certaines colonnes sont **supprimées** (`Time`, `Time_Day`, `Time_Min`) pour ne conserver que les informations pertinentes pour la modélisation, comme `Time_Hour`.  
# 


Delta_time = pd.to_timedelta(credit_card_transactions_data['Time'], unit='s')

credit_card_transactions_data['Time_Day'] = (Delta_time.dt.components.days).astype(int)
credit_card_transactions_data['Time_Hour'] = (Delta_time.dt.components.hours).astype(int)
credit_card_transactions_data['Time_Min'] = (Delta_time.dt.components.minutes).astype(int)

credit_card_transactions_data.drop(['Time', 'Time_Day', 'Time_Min'], axis=1, inplace=True)


# Ce bloc prépare les **features et la cible** pour la modélisation :  
# 
# - `X` contient toutes les colonnes **sauf `Class`**, représentant les **caractéristiques**.  
# - `y` contient uniquement la colonne **`Class`**, qui est la **variable cible** indiquant si une transaction est **frauduleuse (1) ou non (0)**.  
# 
# Cette séparation est essentielle pour **entraîner et évaluer les modèles de machine learning**.
# 


X = credit_card_transactions_data.drop(["Class"], axis=1)
y = credit_card_transactions_data["Class"]

X.tail()

# Cette ligne **sépare le dataset en ensembles d'entraînement et de test** :  
# 
# - `X_train` et `y_train` : **données d'entraînement**, utilisées pour **entraîner le modèle**.  
# - `X_test` et `y_test` : **données de test**, utilisées pour **évaluer les performances** du modèle sur des données non vues.  
# - `test_size=0.2` signifie que **20 % des données** sont réservées pour le test.  
# - `random_state=100` garantit que la **séparation est reproductible**.
# 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Ce bloc crée des **histogrammes superposés pour chaque feature** afin de comparer les distributions entre les transactions **légitimes** et **frauduleuses** :  
# 
# - `cols` contient la liste de toutes les colonnes/features.  
# - `legit_records` et `fraud_records` sont des **masques booléens** pour séparer les transactions non frauduleuses et frauduleuses.  
# - Pour chaque colonne, `sns.distplot` trace :  
#   - En **vert** : la distribution des transactions légitimes.  
#   - En **rouge** : la distribution des transactions frauduleuses.  
# - Les sous-graphes (`plt.subplot`) permettent de **visualiser toutes les features dans une seule figure**.  
# 
# Cette visualisation aide à **identifier quelles variables peuvent mieux différencier les fraudes des transactions normales**.
# 


cols = list(X.columns.values)

legit_records = credit_card_transactions_data.Class == 0
fraud_records = credit_card_transactions_data.Class == 1

plt.figure(figsize=(20, 60))
for n, col in enumerate(cols):
    plt.subplot(10,3,n+1)
    sns.distplot(X[col][legit_records], color='green')
    sns.distplot(X[col][fraud_records], color='red')
    plt.title(col, fontsize=17)

plt.show()
    

# Cette ligne crée un **DataFrame vide nommé `df_Results`** pour **stocker les résultats des différents modèles** testés dans le projet.  
# 
# - Les colonnes du DataFrame sont :  
#   - **`Methodology`** : la méthode ou approche utilisée.  
#   - **`Model`** : le nom du modèle de machine learning.  
#   - **`Accuracy`** : la précision obtenue sur les données de test.  
#   - **`roc_value`** : la valeur de l'AUC-ROC pour évaluer la capacité du modèle à distinguer les classes.  
#   - **`threshold`** : le seuil choisi pour classer une transaction comme frauduleuse ou non.  
# 
# Ce DataFrame servira à **comparer facilement les performances de tous les modèles expérimentés**.
# 


df_Results = pd.DataFrame(columns=['Methodology', 'Model', 'Accuracy', 'roc_value', 'threshold'])

# Cette fonction **`buildAndRunLogisticModels`** permet de **construire, entraîner et évaluer des modèles de régression logistique** avec régularisation L1 et L2 sur le dataset de transactions :  
# 
# - **Hyperparamètres et validation croisée** :  
#   - `num_C` définit une série de valeurs pour le paramètre de régularisation C.  
#   - `KFold` est utilisé pour la **validation croisée à 10 plis**.  
# - **Modèles** :  
#   - `LogisticRegressionCV` avec **L1** (lasso) et **L2** (ridge) pour gérer la régularisation.  
# - **Entraînement** : les deux modèles sont **ajustés sur les données d'entraînement**.  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrices de confusion**, **classification report** et **valeurs ROC-AUC**.  
#   - Calcul du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé des **courbes ROC** pour visualiser la performance sur les données de test.  
# - **Stockage des résultats** :  
#   - Les performances de chaque modèle (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison ultérieure.  
# 
# Cette fonction permet donc de **tester et comparer efficacement deux approches de régression logistique** sur des données déséquilibrées.
# 


def buildAndRunLogisticModels(df_Results, Methodology, X_train, y_train, X_test, y_test):

    num_C = list(np.power(10.0, np.arange(-10, 10)))
    cv_num = KFold(n_splits=10, shuffle=True, random_state=42)

    searchCV_l2 = LogisticRegressionCV(
        Cs=num_C, penalty='l2', scoring='roc_auc', cv=cv_num,
        random_state=42, max_iter=10000, fit_intercept=True,
        solver='newton-cg', tol=10
    )

    searchCV_l1 = LogisticRegressionCV(
        Cs=num_C, penalty='l1', scoring='roc_auc', cv=cv_num,
        random_state=42, max_iter=10000, fit_intercept=True,
        solver='liblinear', tol=10
    )

    # ==================== MLflow RUN pour L2 ====================
    with mlflow.start_run(run_name=f"Logistic_L2_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        searchCV_l2.fit(X_train, y_train)
        y_pred_l2 = searchCV_l2.predict(X_test)
        y_pred_probs_l2 = searchCV_l2.predict_proba(X_test)[:, 1]

        Accuracy_l2 = metrics.accuracy_score(y_pred=y_pred_l2, y_true=y_test)
        roc_value_l2 = metrics.roc_auc_score(y_test, y_pred_probs_l2)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs_l2)
        threshold_l2 = thresholds[np.argmax(tpr - fpr)]

        mlflow.log_param("Model", "Logistic Regression")
        mlflow.log_param("Regularization", "L2")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_param("Best_C", searchCV_l2.C_[0])
        mlflow.log_metric("Accuracy", Accuracy_l2)
        mlflow.log_metric("ROC_AUC", roc_value_l2)
        mlflow.log_metric("Threshold", threshold_l2)

        mlflow.sklearn.log_model(searchCV_l2, artifact_path="model")

        df_Results = pd.concat([df_Results, pd.DataFrame({
            'Methodology': [Methodology],
            'Model': ['Logistic Regression with L2 Regularization'],
            'Accuracy': [Accuracy_l2],
            'roc_value': [roc_value_l2],
            'threshold': [threshold_l2]
        })], ignore_index=True)

    # ==================== MLflow RUN pour L1 ====================
    with mlflow.start_run(run_name=f"Logistic_L1_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        searchCV_l1.fit(X_train, y_train)
        y_pred_l1 = searchCV_l1.predict(X_test)
        y_pred_probs_l1 = searchCV_l1.predict_proba(X_test)[:, 1]

        Accuracy_l1 = metrics.accuracy_score(y_pred=y_pred_l1, y_true=y_test)
        roc_value_l1 = metrics.roc_auc_score(y_test, y_pred_probs_l1)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs_l1)
        threshold_l1 = thresholds[np.argmax(tpr - fpr)]

        # Log paramètres et métriques
        mlflow.log_param("Model", "Logistic Regression")
        mlflow.log_param("Regularization", "L1")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_param("Best_C", searchCV_l1.C_[0])
        mlflow.log_metric("Accuracy", Accuracy_l1)
        mlflow.log_metric("ROC_AUC", roc_value_l1)
        mlflow.log_metric("Threshold", threshold_l1)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(searchCV_l1, artifact_path="model")

        df_Results = pd.concat([df_Results, pd.DataFrame({
            'Methodology': [Methodology],
            'Model': ['Logistic Regression with L1 Regularization'],
            'Accuracy': [Accuracy_l1],
            'roc_value': [roc_value_l1],
            'threshold': [threshold_l1]
        })], ignore_index=True)

    return df_Results


# Cette fonction **`buildAndRunKNNModels`** permet de **construire, entraîner et évaluer un modèle K-Nearest Neighbors (KNN)** sur le dataset de transactions :  
# 
# - **Modèle KNN** :  
#   - `KNeighborsClassifier` avec **5 voisins** et parallélisation (`n_jobs=16`) pour accélérer l'entraînement.  
# - **Entraînement** : le modèle est ajusté sur les **données d'entraînement** (`X_train`, `y_train`).  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrice de confusion** et **classification report** sur l'ensemble de test.  
#   - Calcul des **probabilités prédites** pour la classe positive (fraude).  
#   - Calcul de la **valeur ROC-AUC** et du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé de la **courbe ROC** pour visualiser la performance du modèle.  
# - **Stockage des résultats** : les performances du modèle (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison avec d'autres modèles.  
# 
# Cette fonction fournit une **évaluation complète du KNN** dans le contexte de la détection de fraude.
# 


def buildAndRunKNNModels(df_Results, Methodology, X_train, y_train, X_test, y_test):

    # ==================== Démarrage du tracking MLflow ====================
    with mlflow.start_run(run_name=f"KNN_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        
        # --- Entraînement du modèle ---
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=16)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        print("Model score :", score)

        # --- Prédictions et évaluation ---
        y_pred = knn.predict(X_test)
        KNN_Accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
        print("Classification Report:", metrics.classification_report(y_test, y_pred))

        knn_probs = knn.predict_proba(X_test)[:, 1]
        knn_roc_value = metrics.roc_auc_score(y_test, knn_probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, knn_probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        roc_auc = metrics.auc(fpr, tpr)

        print(f"KNN ROC value: {knn_roc_value:.4f}")
        print(f"Optimal threshold: {threshold:.4f}")
        plt.plot(fpr, tpr, label="Test, auc=" + str(round(roc_auc, 4)))
        plt.legend(loc=4)
        plt.show()

        # --- Log des paramètres ---
        mlflow.log_param("Model", "KNN")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_param("n_neighbors", 5)
        mlflow.log_param("n_jobs", 16)

        # --- Log des métriques ---
        mlflow.log_metric("Accuracy", KNN_Accuracy)
        mlflow.log_metric("ROC_AUC", knn_roc_value)
        mlflow.log_metric("Threshold", threshold)

        # --- Enregistrement du modèle dans MLflow ---
        mlflow.sklearn.log_model(knn, artifact_path="model")

        # --- Enregistrement des résultats dans le DataFrame ---
        df_Results = pd.concat([
            df_Results,
            pd.DataFrame({
                'Methodology': [Methodology],
                'Model': ['KNN'],
                'Accuracy': [KNN_Accuracy],
                'roc_value': [knn_roc_value],
                'threshold': [threshold]
            })
        ], ignore_index=True)

    return df_Results


# Cette fonction **`buildAndRunTreeModels`** permet de **construire, entraîner et évaluer des modèles d'arbre de décision** sur le dataset de transactions :  
# 
# - **Critères d'arbre** : `gini` et `entropy` sont testés pour mesurer la **pureté des nœuds** lors de la construction de l'arbre.  
# - **Entraînement** : chaque arbre est ajusté sur les **données d'entraînement** (`X_train`, `y_train`).  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrice de confusion** et **classification report** sur l'ensemble de test.  
#   - Calcul des **probabilités prédites** pour la classe positive (fraude).  
#   - Calcul de la **valeur ROC-AUC** et du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé de la **courbe ROC** pour visualiser la performance du modèle pour chaque critère.  
# - **Stockage des résultats** : les performances de chaque arbre (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison avec d'autres modèles.  
# 
# Cette fonction permet ainsi de **comparer facilement l'impact du critère choisi sur les performances de l'arbre de décision**.
# 


def buildAndRunTreeModels(df_Results, Methodology, X_train, y_train, X_test, y_test):
    criteria = ['gini', 'entropy']

    for c in criteria:
        with mlflow.start_run(run_name=f"DecisionTree_{c}_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
            # --- Entraînement du modèle ---
            dt = DecisionTreeClassifier(criterion=c, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            test_score = dt.score(X_test, y_test)
            tree_probs = dt.predict_proba(X_test)[:, 1]
            tree_roc_value = metrics.roc_auc_score(y_test, tree_probs)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, tree_probs)
            threshold = thresholds[np.argmax(tpr - fpr)]
            roc_auc = metrics.auc(fpr, tpr)

            # --- Affichage des résultats ---
            print(f"{c} score : {test_score:.4f}")
            print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
            print("Classification Report:", metrics.classification_report(y_test, y_pred))
            print(f"ROC AUC: {tree_roc_value:.4f} | Threshold: {threshold:.4f}")
            plt.plot(fpr, tpr, label=f"Test ({c}), auc={roc_auc:.4f}")
            plt.legend(loc=4)
            plt.show()

            # --- Log dans MLflow ---
            mlflow.log_param("Model", "Decision Tree")
            mlflow.log_param("Criterion", c)
            mlflow.log_param("Methodology", Methodology)
            mlflow.log_metric("Accuracy", test_score)
            mlflow.log_metric("ROC_AUC", tree_roc_value)
            mlflow.log_metric("Threshold", threshold)

            # Sauvegarde du modèle
            mlflow.sklearn.log_model(dt, artifact_path="model")

            # Enregistrement du résultat dans ton DataFrame
            df_Results = pd.concat([
                df_Results,
                pd.DataFrame({
                    'Methodology': [Methodology],
                    'Model': [f'Decision Tree ({c})'],
                    'Accuracy': [test_score],
                    'roc_value': [tree_roc_value],
                    'threshold': [threshold]
                })
            ], ignore_index=True)

    return df_Results


# Cette fonction **`buildAndRunRandomForestModels`** permet de **construire, entraîner et évaluer un modèle Random Forest** sur le dataset de transactions :  
# 
# - **Modèle Random Forest** :  
#   - `RandomForestClassifier` avec **100 arbres**, bootstrap activé et sélection aléatoire des features (`max_features='sqrt'`).  
# - **Entraînement** : le modèle est ajusté sur les **données d'entraînement** (`X_train`, `y_train`).  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrice de confusion** et **classification report** sur l'ensemble de test.  
#   - Calcul des **probabilités prédites** pour la classe positive (fraude).  
#   - Calcul de la **valeur ROC-AUC** et du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé de la **courbe ROC** pour visualiser la performance du modèle.  
# - **Stockage des résultats** : les performances du modèle (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison avec d'autres modèles.  
# 
# Cette fonction fournit une **évaluation complète du Random Forest** dans le contexte de la détection de fraude.
# 


def buildAndRunRandomForestModels(df_Results, Methodology, X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name=f"RandomForest_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        # --- Entraînement du modèle ---
        RF_model = RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        RF_model.fit(X_train, y_train)
        RF_test_score = RF_model.score(X_test, y_test)
        RF_predictions = RF_model.predict(X_test)

        # --- Évaluation ---
        print(f"Model Accuracy : {RF_test_score:.4f}")
        print("Confusion Matrix :\n", metrics.confusion_matrix(y_test, RF_predictions))
        print("Classification Report :\n", metrics.classification_report(y_test, RF_predictions))

        rf_probs = RF_model.predict_proba(X_test)[:, 1]
        rf_roc_value = metrics.roc_auc_score(y_test, rf_probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, rf_probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        roc_auc = metrics.auc(fpr, tpr)

        print(f"RF ROC AUC: {rf_roc_value:.4f}")
        print(f"RF Threshold: {threshold:.4f}")
        print("ROC for test dataset:", '{:.1%}'.format(roc_auc))

        plt.plot(fpr, tpr, label=f"Test, AUC={roc_auc:.4f}")
        plt.legend(loc=4)
        plt.show()

        # --- Log dans MLflow ---
        mlflow.log_param("Model", "Random Forest")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_features", "sqrt")
        mlflow.log_metric("Accuracy", RF_test_score)
        mlflow.log_metric("ROC_AUC", rf_roc_value)
        mlflow.log_metric("Threshold", threshold)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(RF_model, artifact_path="model")

        # --- Sauvegarde dans le DataFrame local ---
        df_Results = pd.concat([
            df_Results,
            pd.DataFrame({
                'Methodology': [Methodology],
                'Model': ['Random Forest'],
                'Accuracy': [RF_test_score],
                'roc_value': [rf_roc_value],
                'threshold': [threshold]
            })
        ], ignore_index=True)

    return df_Results


# Cette fonction **`buildAndRunXGBoostModels`** permet de **construire, entraîner et évaluer un modèle XGBoost** sur le dataset de transactions :  
# 
# - **Modèle XGBoost** :  
#   - `XGBClassifier` avec les paramètres par défaut et `random_state=42` pour la reproductibilité.  
# - **Entraînement** : le modèle est ajusté sur les **données d'entraînement** (`X_train`, `y_train`).  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrice de confusion** et **classification report** sur l'ensemble de test.  
#   - Calcul des **probabilités prédites** pour la classe positive (fraude).  
#   - Calcul de la **valeur ROC-AUC** et du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé de la **courbe ROC** pour visualiser la performance du modèle.  
# - **Stockage des résultats** : les performances du modèle (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison avec d'autres modèles.  
# 


def buildAndRunXGBoostModels(df_Results, Methodology, X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name=f"XGBoost_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        # --- Entraînement du modèle ---
        XGBmodel = XGBClassifier(random_state=42, n_jobs=-1)
        XGBmodel.fit(X_train, y_train)
        y_pred = XGBmodel.predict(X_test)

        XGB_test_score = XGBmodel.score(X_test, y_test)

        # --- Affichage ---
        print(f'Model Accuracy: {XGB_test_score:.4f}')
        print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

        XGB_probs = XGBmodel.predict_proba(X_test)[:, 1]
        XGB_roc_value = metrics.roc_auc_score(y_test, XGB_probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, XGB_probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        roc_auc = metrics.auc(fpr, tpr)

        print(f"XGB ROC AUC: {XGB_roc_value:.4f}")
        print(f"XGB Threshold: {threshold:.4f}")
        print("ROC for test dataset:", '{:.1%}'.format(roc_auc))

        plt.plot(fpr, tpr, label=f"Test, AUC={roc_auc:.4f}")
        plt.legend(loc=4)
        plt.show()

        # --- Log dans MLflow ---
        mlflow.log_param("Model", "XGBoost")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_metric("Accuracy", XGB_test_score)
        mlflow.log_metric("ROC_AUC", XGB_roc_value)
        mlflow.log_metric("Threshold", threshold)

        # Sauvegarde du modèle
        mlflow.xgboost.log_model(XGBmodel, artifact_path="model")

        # --- Sauvegarde dans le DataFrame local ---
        df_Results = pd.concat([
            df_Results,
            pd.DataFrame({
                'Methodology': [Methodology],
                'Model': ['XGBoost'],
                'Accuracy': [XGB_test_score],
                'roc_value': [XGB_roc_value],
                'threshold': [threshold]
            })
        ], ignore_index=True)

    return df_Results


# Cette fonction **`buildAndRunSVMModels`** permet de **construire, entraîner et évaluer un modèle SVM (Support Vector Machine)** sur le dataset de transactions :  
# 
# - **Modèle SVM** :  
#   - `SVC` avec le **kernel sigmoid** pour capturer des relations non linéaires, et `random_state=42` pour la reproductibilité.  
# - **Entraînement** : le modèle est ajusté sur les **données d'entraînement** (`X_train`, `y_train`).  
# - **Évaluation** :  
#   - Calcul de **l’accuracy**, **matrice de confusion** et **classification report** sur l'ensemble de test.  
#   - Pour obtenir des probabilités, un second SVM est entraîné avec `probability=True`.  
#   - Calcul de la **valeur ROC-AUC** et du **seuil optimal** pour classifier une transaction comme frauduleuse.  
#   - Tracé de la **courbe ROC** pour visualiser la performance du modèle.  
# - **Stockage des résultats** : les performances du modèle (Accuracy, ROC-AUC, seuil) sont ajoutées au **DataFrame `df_Results`** pour comparaison avec d'autres modèles.  
# 


def buildAndRunSVMModels(df_Results, Methodology, X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name=f"SVM_{Methodology}_{datetime.now().strftime('%H%M%S')}"):
        # --- Entraînement du modèle ---
        clf = SVC(kernel='sigmoid', probability=True, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        SVM_score = metrics.accuracy_score(y_test, y_pred)

        # --- Affichage ---
        print(f'Accuracy score: {SVM_score:.4f}')
        print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

        svm_probs = clf.predict_proba(X_test)[:, 1]
        SVM_roc_value = metrics.roc_auc_score(y_test, svm_probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, svm_probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        roc_auc = metrics.auc(fpr, tpr)

        print(f"SVM ROC AUC: {SVM_roc_value:.4f}")
        print(f"SVM Threshold: {threshold:.4f}")
        print("ROC for test dataset:", '{:.1%}'.format(roc_auc))

        plt.plot(fpr, tpr, label=f"Test, AUC={roc_auc:.4f}")
        plt.legend(loc=4)
        plt.show()

        # --- Log dans MLflow ---
        mlflow.log_param("Model", "SVM")
        mlflow.log_param("Methodology", Methodology)
        mlflow.log_param("Kernel", "sigmoid")
        mlflow.log_metric("Accuracy", SVM_score)
        mlflow.log_metric("ROC_AUC", SVM_roc_value)
        mlflow.log_metric("Threshold", threshold)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # --- Sauvegarde dans le DataFrame local ---
        df_Results = pd.concat([
            df_Results,
            pd.DataFrame({
                'Methodology': [Methodology],
                'Model': ['SVM'],
                'Accuracy': [SVM_score],
                'roc_value': [SVM_roc_value],
                'threshold': [threshold]
            })
        ], ignore_index=True)

    return df_Results


# Ce bloc utilise la **validation croisée répétée (Repeated K-Fold)** pour évaluer la robustesse des modèles :  
# 
# - `RepeatedKFold` divise les données en **5 plis** et répète la procédure **10 fois**.  
# - Pour chaque itération :  
#   - `train_index` et `test_index` contiennent les indices des **données d'entraînement** et **de test**.  
#   - `X_train_cv` et `X_test_cv` contiennent les **features** pour l'entraînement et le test.  
#   - `y_train_cv` et `y_test_cv` contiennent les **cibles** correspondantes.  
# - L’impression des indices montre quelles observations sont utilisées pour l’entraînement et le test à chaque pli.  
# 
# Cette technique permet de **réduire la variance des estimations de performance** et d’**évaluer les modèles de manière plus fiable** sur des données déséquilibrées.
# 


rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
for train_index, test_index in rkf.split(X,y):
    print("TRAIN:", train_index, "TEST", test_index)
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

# -
# 
# ### 1️⃣ **Logistic Regression**
# 
# * La régression logistique est exécutée avec deux régularisations : **L1** et **L2**.
# * Le code calcule :
# 
#   * **Accuracy**
#   * **Confusion Matrix**
#   * **Classification Report**
#   * **ROC-AUC et seuil optimal**
# * Résultat clé :
# 
#   * L1 a un **roc\_value = 0.88**, L2 a **roc\_value = 0.57**, donc L1 est beaucoup meilleur pour ce jeu de données.
# * Temps d’exécution : \~126 secondes.
# 
# ---
# 
# ### 2️⃣ **KNN**
# 
# * K-Nearest Neighbors avec `n_neighbors=5`.
# * Mesure de performance : Accuracy, ROC-AUC, seuil optimal.
# * Résultats :
# 
#   * Accuracy ≈ 0.999
#   * ROC ≈ 0.876
# * Temps d’exécution : \~239 secondes.
# 
# ---
# 
# ### 3️⃣ **Decision Tree**
# 
# * Deux critères testés : **gini** et **entropy**.
# * Mesure de performance : Accuracy, ROC-AUC, seuil optimal.
# * Résultats :
# 
#   * gini : ROC ≈ 0.881
#   * entropy : ROC ≈ 0.876
# * Temps d’exécution : \~54 secondes.
# 
# ---
# 
# ### 4️⃣ **Random Forest**
# 
# * Modèle avec 100 arbres, `max_features='sqrt'`.
# * Résultats :
# 
#   * Accuracy ≈ 0.9995
#   * ROC ≈ 0.937
# * Temps d’exécution : \~352 secondes.
# 
# ---
# 
# ### 5️⃣ **XGBoost**
# 
# * Modèle boosting avec `XGBClassifier`.
# * Résultats :
# 
#   * Accuracy ≈ 0.9995
#   * ROC ≈ 0.978 → Meilleur modèle ROC de tous.
# * Temps d’exécution : \~4.6 secondes (très rapide pour XGBoost).
# 
# ---
# 
# ### ✅ **Résumé général**
# 
# * **Meilleur ROC** : XGBoost (97.8%)
# * **Plus rapide** : XGBoost (\~4s), malgré Random Forest plus lent (\~352s).
# * **Seuil optimal** : calculé via `tpr-fpr` pour tous les modèles afin de maximiser le ROC.
# * Ce bloc montre comment **exécuter plusieurs modèles, mesurer per, ROC et temps** pour que ce soit plus lisible.
# 
# Veux‑tu que je fasse ça ?
# 


print("Logistic Regression with 11 and 12 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results, "RepeatedKFold Cross Validation", X_train_cv, y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("KNN Model")
start_time = time.time()
df_Results = buildAndRunKNNModels(df_Results, "RepeatedKFold Cross Validation", X_train_cv, y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("Decision Tree Models with 'gini' and 'entropy' criteria")
start_time = time.time()
df_Results = buildAndRunTreeModels(df_Results, "RepeatedKFold Cross Validation", X_train_cv, y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("Random Forest Model")
start_time = time.time()
df_Results = buildAndRunRandomForestModels(df_Results, "RepeatedKFold Cross Validation", X_train_cv, y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("XGBoost Model")
start_time = time.time()
df_Results = buildAndRunXGBoostModels(df_Results, "RepeatedKFold Cross Validation", X_train_cv, y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )


df_Results

# Ce bloc utilise la **validation croisée stratifiée (Stratified K-Fold)** pour maintenir la **proportion des classes** dans chaque pli :  
# 
# - `StratifiedKFold` divise les données en **5 plis**, en conservant la même répartition des classes dans les ensembles d'entraînement et de test.  
# - Pour chaque pli :  
#   - `train_index` et `test_index` contiennent les indices des **données d'entraînement** et **de test**.  
#   - `X_train_SKF_cv` et `X_test_SKF_cv` contiennent les **features** correspondantes.  
#   - `y_train_SKF_cv` et `y_test_SKF_cv` contiennent les **cibles** correspondantes.  
# - L’impression des indices montre quelles observations sont utilisées pour l’entraînement et le test à chaque pli.  
# 
# Cette approche est particulièrement utile pour les datasets **déséquilibrés**, comme celui de la détection de fraude, afin de **préserver la proportion des fraudes** dans chaque pli.
# 


skf = StratifiedKFold(n_splits=5, random_state=None)
for train_index, test_index in skf.split(X,y):
    print("TRAIN:", train_index, "TEST", test_index)
    X_train_SKF_cv, X_test_SKF_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_SKF_cv, y_test_SKF_cv = y.iloc[train_index], y.iloc[test_index]

# 
# #### **Logistic Regression (L1 & L2 Regularisation)**
# 
# * L1 (lasso) : meilleur ROC AUC = 0.889 → seuil optimal ≈ 0.021.
# * L2 (ridge) : ROC AUC faible = 0.611 → seuil ≈ 0.499.
# * L1 gère mieux la détection de la classe minoritaire.
# * Accuracy global très élevée (\~0.9987 pour L1).
# 
# #### **KNN**
# 
# * Accuracy = 0.9992
# * ROC AUC = 0.806, seuil = 0.2
# * Bonne performance globale, mais recall de la classe minoritaire plus faible que Logistic L1.
# 
# #### **Decision Tree**
# 
# * Critère Gini : Accuracy = 0.9988, ROC AUC = 0.826
# * Critère Entropy : Accuracy = 0.9990, ROC AUC = 0.821
# * Bonnes performances, légèrement inférieures aux modèles d’ensemble.
# 
# #### **Random Forest**
# 
# * Accuracy = 0.9994
# * ROC AUC = 0.946 → seuil très faible = 0.01
# * Très bon équilibre entre précision et rappel pour la classe minoritaire.
# 
# #### **XGBoost**
# 
# * Accuracy = 0.9994
# * ROC AUC = 0.972 → seuil très faible ≈ 3.77e-5
# * Meilleure performance globale, surtout pour la détection des cas minoritaires.
# * Temps d’entraînement très rapide par rapport aux autres modèles d’ensemble.


print("Logistic Regression with 11 and 12 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results, "StratifiedKFold Cross Validation", X_train_SKF_cv, y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("KNN Model")
start_time = time.time()
df_Results = buildAndRunKNNModels(df_Results, "StratifiedKFold Cross Validation", X_train_SKF_cv, y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("Decision Tree Models with 'gini' and 'entropy' criteria")
start_time = time.time()
df_Results = buildAndRunTreeModels(df_Results, "StratifiedKFold Cross Validation", X_train_SKF_cv, y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("Random Forest Model")
start_time = time.time()
df_Results = buildAndRunRandomForestModels(df_Results, "StratifiedKFold Cross Validation", X_train_SKF_cv, y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

print("XGBoost Model")
start_time = time.time()
df_Results = buildAndRunXGBoostModels(df_Results, "StratifiedKFold Cross Validation", X_train_SKF_cv, y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---"% ( time.time() - start_time))
print('-'*60 )

df_Results 


param_test = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'n_estimators':range(60,130,150),
    'learning_rate':[0.05,0.1,0.125,0.15,0.2],
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(7,10)],
    'colsample_bytree':[i/10.0 for i in range(7,10)]
}

gsearch1 = RandomizedSearchCV(estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                        colsample_bynode=1,max_delta_step=0,
                                                        missing=None, n_jobs=-1,
                                                        nthread=None, objective='binary:logistic', random_state=42,
                                                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                        silent=None, verbosity=1),
                                                        param_distributions = param_test, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5)
gsearch1.fit(X_train_cv, y_train_cv)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# Enfin, nous définissons et entraînons le **classifieur XGBoost final** avec les meilleurs hyperparamètres trouvés lors de la recherche aléatoire.  
# 
# - **Configuration du modèle** :  
#   - `max_depth=9` et `min_child_weight=1` pour gérer la complexité des arbres.  
#   - `n_estimators=60` et `learning_rate=0.2` pour contrôler le nombre d’arbres et la vitesse d’apprentissage.  
#   - `subsample=0.7` et `colsample_bytree=0.9` pour réduire le surapprentissage.  
#   - `gamma=0.3` pour régulariser et éviter les branches inutiles.  
#   - `objective='binary:logistic'` car il s’agit d’un problème de classification binaire (fraude ou non).  
# 
# - **Entraînement** : le modèle est entraîné sur les données **oversamplées (X_over, y_over)** pour mieux gérer le déséquilibre des classes.  
# - **Évaluation** :  
#   - `XGB_test_score` : précision globale sur l’ensemble de test.  
#   - `XGB_probs` : probabilités prédites pour la classe positive (fraude).  
#   - `XGB_roc_value` : score ROC-AUC, mesurant la capacité du modèle à séparer les classes.  
#   - `optimal_threshold` : seuil optimal calculé à partir de la courbe ROC pour maximiser la sensibilité et la spécificité.
#     
#   
#   - **Model accuracy** : 0.9994  
#   - **XGB ROC-AUC** 0.9807  
#   - **Seuil optimal** : 0.0005  
# 
# Ce modèle XGBoost montre donc une **excellente capacité à détecter les transactions frauduleuses** dans notre dataset.
# 


# Define the XGBoost classifier with updated parameters
clf = XGBClassifier(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=0.9,
    gamma=0.3,
    learning_rate=0.2,
    max_delta_step=0,
    max_depth=9,
    min_child_weight=1,
    n_estimators=60,
    n_jobs=1,
    objective='binary:logistic',
    random_state=42,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=0.7,
    verbosity=1
)

# Train the classifier
clf.fit(X_train_cv, y_train_cv)

# Evaluate accuracy on the test set
XGB_test_score = clf.score(X_test_cv, y_test_cv)
print(f"Model accuracy: {XGB_test_score:.4f}")

# Get predicted probabilities for the positive class
XGB_probs = clf.predict_proba(X_test_cv)[:, 1]

# Calculate the ROC-AUC score
XGB_roc_value = metrics.roc_auc_score(y_test_cv, XGB_probs)
print(f"XGB ROC-AUC value: {XGB_roc_value:.4f}")

# Calculate ROC curve and find the optimal threshold
fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, XGB_probs)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"XGB optimal threshold: {optimal_threshold:.4f}")

# Démarrer une nouvelle expérience MLflow
with mlflow.start_run(run_name=f"XGBOOST_Final_Result_{datetime.now().strftime('%H%M%S')}"):
    # Enregistrer les paramètres du modèle
    mlflow.log_params(clf.get_params())

    # Enregistrer le modèle XGBoost
    mlflow.xgboost.log_model(clf, "model")

    # Enregistrer les métriques
    mlflow.log_metric("accuracy", XGB_test_score)
    mlflow.log_metric("roc_auc", XGB_roc_value)
    mlflow.log_metric("optimal_threshold", optimal_threshold)


df_Results.to_csv("results.csv", index=False)
mlflow.log_artifact("results.csv")

import mlflow.pyfunc

best_model_uri = "runs:/cec76408088449a09313e4f7f1e1335c/model"
model_details = mlflow.register_model(best_model_uri, "CreditCardFraudModel")


from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="CreditCardFraudModel",
    version=model_details.version,
    stage="Production"
)