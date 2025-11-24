# Supervised Learning - Prédiction du Résultat Net des Entreprises Françaises

**Contributeurs :** Alaphilippe Thomas, Gilhodes Nicolas, Zanchi Ugo

---

## 📋 Description du Business Challenge

### Contexte
Les défaillances d'entreprises représentent un enjeu économique majeur en France. Anticiper la santé financière des entreprises permet aux banques, investisseurs et décideurs publics de mieux allouer leurs ressources et de prévenir les risques.

### Objectif
Prédire le **résultat net (N+1)** des entreprises françaises à partir de leurs données démographiques (SIRENE) et financières (bilans INPI), permettant ainsi d'identifier en amont les entreprises à risque et celles en bonne santé financière.

### Parties Prenantes
- **Banques & Institutions financières** : évaluation du risque de crédit
- **Investisseurs** : aide à la décision d'investissement
- **Organismes publics** : prévention des défaillances d'entreprises

### Impact Mesurable
- Réduction des créances douteuses
- Optimisation de l'allocation de capital
- Identification précoce des signaux de détresse financière

---

## 📊 Dataset

### Sources de Données
Le dataset combine deux sources officielles françaises :

1. **Base SIRENE (INSEE)** : données démographiques des entreprises
   - URL : [data.gouv.fr - SIRENE]([https://www.data.gouv.fr](https://www.data.gouv.fr/api/1/datasets/r/a29c1297-1f92-4e2a-8f6b-8c902ce96c5f)
   - URL : [data.gouv.fr - SIRENE]([https://www.data.gouv.fr](https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c)
   - URL : [data.gouv.fr - SIRENE]([https://www.data.gouv.fr](https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3)
   - Contenu : SIREN, secteur d'activité (NAF), département, effectifs, date de création

2. **Bilans financiers (INPI)** : comptes annuels des entreprises
   - URL : [data.gouv.fr - INPI]([https://www.data.gouv.fr](https://www.data.gouv.fr/api/1/datasets/r/c4ac8f98-2c97-4417-9070-0cbb9de03875)
   - Contenu : résultat net, actif total, dettes, trésorerie, résultats exceptionnels

### Caractéristiques du Dataset Final
- **Période couverte** : 2016-2022 (7 années de bilans)
- **Nombre d'observations** : ~500 000 entreprises-années
- **Features** : 25+ variables (ratios financiers, variables démographiques, features engineerées)
- **Target** : `Y_RN` (Résultat Net année N+1)

### Construction du Dataset
Le dataset est construit via un pipeline automatisé :
```bash
make download  # Télécharge les données brutes depuis data.gouv.fr
make process   # Exécute les scripts de preprocessing
```

Scripts de preprocessing :
- `01_build_sirene_infos.py` : nettoyage et enrichissement des données SIRENE
- `02_build_sirene_bilan.py` : pivot des bilans comptables et calcul de ratios financiers
- `03_build_sirene_final.py` : fusion et création du dataset ML-ready

---

## 🚀 Instructions de Reproduction

### Prérequis
- **Python** : 3.10+ (testé avec Python 3.14)
- **Espace disque** : ~5 GB pour les données brutes
- **OS** : Linux/macOS (le Makefile utilise bash)

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/ugozchi/AS_Supervised_Learning.git
cd AS_Supervised_Learning
```

2. **Créer l'environnement virtuel et installer les dépendances**
```bash
make setup
```
Cela crée un environnement virtuel `.venv` et installe toutes les dépendances listées dans `requirements.txt`.

3. **Télécharger les données**
```bash
make download
```
Télécharge les 4 fichiers de données brutes depuis data.gouv.fr dans `Data/raw/`.

4. **Préprocesser les données**
```bash
make process
```
Exécute les 3 scripts de preprocessing et génère les fichiers finaux dans `Data/processed/`.

### Lancer l'entraînement

```bash
source .venv/bin/activate
python main.py
```

Le script `main.py` :
1. Charge le dataset depuis `Data/processed/sirene_final2.parquet`
2. Applique le feature engineering
3. Effectue un split temporel train/test (80/20)
4. Entraîne deux modèles XGBoost :
   - **Classification** : prédiction du signe (profit/perte)
   - **Régression** : prédiction de la magnitude (valeur absolue)
5. Combine les prédictions et affiche les métriques
6. Génère les visualisations dans `dashboard_final.png`

**Durée d'exécution** : ~15-20 minutes sur machine standard

---

## 📈 Baseline & Résultats

### Baseline (Version 1)

**Modèle** : XGBoost avec architecture hybride (Classification + Régression)

**Features** :
- Ratios financiers bruts (7 ratios : rentabilité, endettement, trésorerie, etc.)
- Variables démographiques (secteur NAF, département, ancienneté)
- Transformations log des variables financières
- Taux de croissance du résultat net (N-1, N-2)

**Preprocessing** :
- Target Encoding pour les variables catégorielles
- Split temporel 80/20 (train 2016-2020, test 2021-2022)
- Winsorisation des outliers (2.5%-97.5%)

**Hyperparamètres XGBoost** :
- Classification : 700 estimators, lr=0.03, max_depth=7
- Régression : 1500 estimators, lr=0.03, max_depth=6, reg_alpha/lambda=0.5

**Métriques Baseline** :
| Métrique | Valeur |
|----------|--------|
| **MAE** | 352 099 € |
| **RMSE** | 1 785 000 € |
| **R²** | 0.5644 |
| **MAPE** | 181.58% |
| **MedAE** | 48 500 € |

---

### Experiment Tracking

#### Expérience 1 : Baseline
- **Changements** : Architecture hybride Classification + Régression
- **Résultats** : R²=0.5644, MAE=352k€
- **Observation** : Bonnes performances sur les profits, difficultés sur les pertes

#### Expérience 2 : Nettoyage des ratios aberrants
- **Changements** : Clipping des ratios entre -10 et +10
- **Résultats** : R²=0.570 (+0.6%), MAE=348k€ (-1.2%)
- **Observation** : Légère amélioration de la stabilité

#### Expérience 3 : Ajout feature `flag_dettes_explosives`
- **Changements** : Flag binaire si ratio_endettement > 2.0
- **Résultats** : R²=0.572 (+1.3%), MAE=345k€ (-2.0%)
- **Observation** : Meilleure identification des entreprises en difficulté

#### Expérience 4 : Optimisation hyperparamètres (régularisation)
- **Changements** : Augmentation reg_alpha/lambda, ajustement subsample
- **Résultats** : R²=0.568 (-0.7%), MAE=350k€
- **Observation** : Sur-régularisation, retour à la version précédente

**Meilleure version finale** : Expérience 3
- **R² = 0.572** : excellent pour des données financières réelles
- **MAE = 345k€** : dans la norme du secteur (médiane à 48k€)
- **Interprétabilité** : features simples et explicables

---

## 📁 Structure du Repository

```
.
├── README.md                    # Ce fichier
├── main.py                      # Pipeline d'entraînement principal
├── Notebooks/
│   ├── 00_Sandbox.ipynb
│   └── 01_EDA.ipynb
├── requirements.txt             # Dépendances Python
├── Makefile                     # Automatisation du pipeline
├── Scripts/
│   ├── 01_build_sirene_infos.py    # Preprocessing SIRENE
│   ├── 02_build_sirene_bilan.py    # Preprocessing bilans INPI
│   └── 03_build_sirene_final.py    # Fusion finale
├── Data/
│   ├── raw/                     # Données brutes (téléchargées)
│   └── processed/               # Données preprocessées
│       ├── sirene_infos.parquet
│       ├── sirene_bilan.parquet
│       └── sirene_final2.parquet
└── dashboard_final.png          # Visualisations (généré par main.py)
```

---

## 🛠️ Technologies Utilisées

- **Python 3.14**
- **Polars** : manipulation de données haute performance
- **XGBoost** : modèles de gradient boosting
- **scikit-learn** : preprocessing, métriques, validation
- **category-encoders** : Target Encoding
- **pandas, numpy** : calculs numériques
- **matplotlib, seaborn** : visualisations

---

## 💡 Conclusions & Perspectives

### Points Forts
- ✅ R² de **0.572** est excellent pour de la prédiction financière réelle
- ✅ Architecture hybride efficace pour gérer signe et magnitude
- ✅ Pipeline reproductible et automatisé via Makefile
- ✅ Features interprétables (ratios financiers standards)

### Limitations
- ⚠️ Prédiction des pertes plus difficile (asymétrie du problème)
- ⚠️ MAPE élevé sur les petites valeurs (sensibilité aux valeurs proches de 0)
- ⚠️ Absence de données macroéconomiques (taux d'intérêt, PIB)

### Améliorations Futures
1. **Features temporelles** : moyennes mobiles, tendances sur 3 ans
2. **Données externes** : indicateurs sectoriels, données macroéconomiques
3. **Modèles avancés** : LightGBM, CatBoost, ensembles de modèles
4. **Segmentation** : modèles spécialisés par secteur d'activité
5. **API de déploiement** : servir le modèle via FastAPI (bonus)
