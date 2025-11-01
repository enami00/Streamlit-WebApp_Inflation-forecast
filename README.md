[README.md](https://github.com/user-attachments/files/23285949/README.md)
# 📈 Projet : Prévision de l’Inflation

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Objectif du projet

Ce projet vise à **analyser, modéliser et prévoir l’inflation** à partir de **données macroéconomiques trimestrielles**.  
Les données proviennent du site officiel [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/).

L’objectif principal est de fournir un **outil interactif via Streamlit** permettant :
- d’évaluer l’impact des facteurs économiques clés ;
- et de **projeter l’inflation globale (IPC YoY)** sur un horizon de **8 trimestres (N+2)**.

---

## 🧱 Structure du projet

Le projet se compose de trois fichiers principaux :

1. **`data_processor.py`**  
   - Prépare et nettoie les données issues du CSV brut.  
   - Calcule les **taux d’inflation annuels (YoY)** pour :
     - l’IPC global  
     - l’IPC hors alimentation et énergie.  
   - Produit un fichier de sortie prêt à l’analyse :  
     ➜ `inflation_data_YoY_quarterly.csv`

2. **`inflation_app.py`**  
   - Contient l’application **Streamlit** et la **modélisation de la régression linéaire multiple**.  
   - Visualise les corrélations, évalue la performance et génère la **prévision N+2**.  

3. **`inflation_data.csv`**  
   - Fichier source contenant les **données macroéconomiques brutes** (10 variables).

---

## ⚙️ Étape 1 — Traitement des données (`data_processor.py`)

Ce script transforme les données brutes en une série temporelle exploitable.

### Fonctionnalités clés :
- 🔹 **Correction des données** : décimales, formats et noms de colonnes.  
- 🔹 **Calcul du taux annuel YoY** pour les indices de prix :  
  > `Taux YoY = ( (IPC_t / IPC_{t-4}) - 1 ) × 100`
- 🔹 **Fichier de sortie** : `inflation_data_YoY_quarterly.csv`

---

## 💻 Étape 2 — Application Streamlit (`inflation_app.py`)

Ce script lance l’interface interactive d’analyse et de prévision.

### 🧠 Modèle économétrique utilisé
Régression Linéaire Multiple (simplifiée de type ARX) :

> `Inflationₜ = β₀ + Σᵢ βᵢ ⋅ Facteur_Macroᵢ + βlag ⋅ Inflationₜ₋ₗ + εₜ`

### Fonctionnalités principales :
- **Cible par défaut** : Inflation Globale (IPC YoY)  
- **Variables explicatives** : facteurs macroéconomiques standardisés  
- **Variable de retard (lag)** : configurable via la barre latérale  
- **Visualisations** :
  - Corrélations entre variables  
  - Graphiques interactifs  
  - Prévisions futures (jusqu’à N+2)

---

## 🔮 Étape 3 — Prévision Séquentielle (N+2)

Le modèle génère une **prévision pour les 8 trimestres suivants**.  
La prédiction est **séquentielle** : la valeur prédite à T+1 sert d’entrée (lag) pour prédire T+2, etc.

---

## 📊 Données utilisées

Le modèle s’appuie sur les variables suivantes (après nettoyage et alignement) :

| Type | Variable | Description |
|------|-----------|-------------|
| 🎯 Cible | `Inflation_YoY_IPC` | Inflation annuelle globale (YoY) |
| 🎯 Cible alt. | `Inflation_YoY_IPC_non_food_non_energy` | Inflation hors énergie et alimentation |
| 📈 Macro | `log_GDP` | PIB (log transformé) |
| 💰 Monétaire | `Interest_rate_-_Bond_yields` | Taux d’intérêt obligataire |
| 👷 Emploi | `Total_unemployment_rate` | Taux de chômage |
| 🛢️ Énergie | `Crude_oil_price_-_Euro` | Prix du pétrole brut |
| 💱 Change | `Xchange_rate_-_US_to_one_EU` | Taux de change €/$ |
| 💶 Monétaire | `Monetary_aggregate_-_Euro_area` | Masse monétaire |
| 🏛️ Public | `Government_spending` | Dépenses publiques |
| 🗓️ Date | `Observation_date` | Date d’observation (trimestre) |

---

## ⚡ Utilisation

1. **Étape 1 :** Exécuter le script de traitement des données  
   ```bash
   python data_processor.py


2. **Étape 2 :** Lancer l’application Streamlit  
   ```bash
   streamlit run inflation_app.py

3. **Étape 3 :**  
   - Sélectionnez les paramètres du modèle (variables, lag, split train/test).  
   - Visualisez :
     - les **corrélations** entre variables  
     - la **performance du modèle**  
     - les **prévisions N+2**

---

## 📈 Résultats attendus

- ✅ Un fichier **`inflation_data_YoY_quarterly.csv`** propre et exploitable.  
- ✅ Une **application interactive Streamlit** permettant :
  - d’analyser les **tendances passées**  
  - d’évaluer la **performance du modèle**  
  - et de **prévoir l’inflation future**

---

## ⚠️ Limites et pistes d’amélioration

Le modèle actuel (Régression Linéaire Multiple) reste **pédagogique** et présente certaines limites :
- Difficulté à capturer les **non-linéarités** et **changements de régime économiques**.  
- Hypothèse simplifiée : les facteurs macroéconomiques restent constants à horizon N+2.

### 🔍 Améliorations possibles :
- **Modélisation avancée** : adopter des modèles plus puissants tels que **ARIMAX**, **VAR**, ou des algorithmes de **Machine Learning** (Random Forest, XGBoost).  
- **Validation statistique** : ajouter des tests de stationnarité (ADF, KPSS) et sélectionner l’ordre de lag optimal via **AIC/BIC**.  
- **Prévisions plus réalistes** : intégrer les **prévisions officielles** des institutions (BCE, FMI) pour les facteurs macroéconomiques.  
- **Enrichissement des données** : inclure des indicateurs d’attentes inflationnistes et de tensions sur l’offre.  
- **Automatisation** : connexion directe aux **APIs BCE** ou **Eurostat** pour des mises à jour automatiques.




   
