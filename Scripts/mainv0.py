import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from datetime import datetime

# ==============================================================================
# 1. Configuration et Chargement des Données
# ==============================================================================

FILE_PATH = 'Data/processed/sirene_infos_CLEAN.parquet' 
# Ajustez cette date pour assurer une bonne répartition (ex: 2017-01-01 ou 2019-01-01)
SPLIT_DATE = '2018-01-01' 
RANDOM_SEED = 42

def load_data(file_path):
    """
    Charge les données, sépare X et y, gère les dates non valides et applique 
    la Feature Engineering APE.
    """
    print(f"Chargement des données depuis : {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{file_path}' est introuvable. Veuillez vérifier le chemin.")
        return None, None
    
    # 1. Variable Cible (y)
    y = df['is_failed_in_3y']

    # 2. Variables Explicatives (X)
    leakage_columns = ['is_failed_in_3y', 'dateFermeture', 'date_limite_3_ans', 'siren']
    X = df.drop(columns=leakage_columns, errors='ignore')

    # 3. Conversion de la colonne de date AVEC gestion des erreurs (OutOfBoundsDatetime)
    if 'dateCreationUniteLegale' in X.columns:
        X['dateCreationUniteLegale'] = pd.to_datetime(
            X['dateCreationUniteLegale'], 
            errors='coerce'  # Convertit les dates non valides (ex: an 0006) en NaT
        )
    
    # --- FEATURE ENGINEERING POUR GÉRER LA LARGEUR DE LA MATRICE ---
    if 'activitePrincipaleUniteLegale' in X.columns:
        # Réduction du code APE/NAF à ses 2 premiers chiffres (niveau 2 d'agrégation)
        X['activitePrincipaleNiveau2'] = X['activitePrincipaleUniteLegale'].astype(str).str[:2].fillna('NA')
    # -------------------------------------------------------------
        
    return X, y

# ==============================================================================
# 2. Split Temporel (Train/Test Split)
# ==============================================================================

def temporal_split(X, y, split_date):
    """
    Effectue un split d'entraînement et de test basé sur la date de création.
    """
    split_dt = pd.to_datetime(split_date)
    
    # Séparation : Train = avant la date; Test = à partir de la date
    train_mask = X['dateCreationUniteLegale'] < split_dt
    
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    
    X_test = X[~train_mask].copy()
    y_test = y[~train_mask].copy()

    if len(X_train) == 0:
        print("\nERREUR CRITIQUE: L'ensemble d'entraînement (X_train) est VIDE.")
        print(f"Veuillez vérifier votre SPLIT_DATE ({split_date}) par rapport à la plage de dates de vos données.")
        return None, None, None, None
    
    print(f"\n--- Split Temporel Effectué (Coupure: {split_date}) ---")
    print(f"Taille du Train: {len(X_train)} entreprises.")
    print(f"Taille du Test: {len(X_test)} entreprises.")
    print(f"Proportion de défaillances (1) dans Train: {y_train.mean():.2%}")
    
    # Retirer la date de création de X_train et X_test après le split
    X_train = X_train.drop(columns=['dateCreationUniteLegale'])
    X_test = X_test.drop(columns=['dateCreationUniteLegale'])

    return X_train, X_test, y_train, y_test

# ==============================================================================
# 3. Pré-traitement des Features (ColumnTransformer)
# ==============================================================================

def build_preprocessor(X_train):
    """
    Définit le ColumnTransformer pour appliquer le pré-processing adéquat.
    Utilise sparse_output=True pour gérer la mémoire.
    """
    
    # Définition des types de colonnes
    numeric_features = ['anneeCreation', 'moisCreation']
    
    # Variables catégorielles (inclut la nouvelle feature agrégée)
    categorical_features = [
        col for col in X_train.columns 
        if col not in numeric_features 
        and col not in ['dateCreationUniteLegale']
        # Exclusion de la feature trop détaillée
        and col != 'activitePrincipaleUniteLegale' 
    ]
    
    # Pipeline pour les variables numériques: Standardisation
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour les variables catégorielles: One-Hot Encoding
    # IMPORTANT: sparse_output=True pour économiser la mémoire.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

# ==============================================================================
# 4. Modèles et Pipelines
# ==============================================================================

def build_model_pipeline(preprocessor, model_name='baseline'):
    """
    Construit et retourne le pipeline complet pour le modèle spécifié.
    Utilise des solvers adaptés aux données creuses et n_jobs=-1 pour la performance.
    """
    if model_name == 'baseline':
        print("\nPipeline du Modèle: Régression Logistique (Baseline)")
        model = LogisticRegression(
            random_state=RANDOM_SEED, 
            # 'saga' est préférable pour les matrices creuses et les grands jeux de données
            solver='saga', 
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1 # Utilisation des cœurs disponibles pour l'accélération
        )
    elif model_name == 'iteration_rf':
        print("\nPipeline du Modèle: Random Forest (Itération 1)")
        model = RandomForestClassifier(
            n_estimators=150,      
            max_depth=15,          
            class_weight='balanced', 
            random_state=RANDOM_SEED,
            n_jobs=-1 
        )
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

# ==============================================================================
# 5. Évaluation et Cross-Validation
# ==============================================================================

def run_cross_validation(pipeline, X_train, y_train, n_splits=5):
    """
    Effectue une Cross-Validation temporelle (TimeSeriesSplit) sur le jeu d'entraînement.
    """
    print(f"\n--- Cross-Validation Temporelle (Splits={n_splits}) ---")
    
    # TimeSeriesSplit est utilisé pour respecter la contrainte temporelle
    tscv = TimeSeriesSplit(n_splits=n_splits) 

    try:
        cv_scores = cross_val_score(
            pipeline, 
            X_train, 
            y_train, 
            cv=tscv, 
            scoring='roc_auc', 
            n_jobs=-1
        )

        print(f"Score ROC AUC CV (Moyenne): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return cv_scores.mean()
    except ValueError as e:
        print(f"Erreur lors de la Cross-Validation. Assurez-vous d'avoir assez de données/splits: {e}")
        return np.nan

def evaluate_model(pipeline, X_test, y_test):
    """
    Calcule et affiche les métriques de classification sur le jeu de test.
    """
    
    # Prédictions et Probabilités
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    metrics = {
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Recall (Rappel)": recall_score(y_test, y_pred, zero_division=0),
        "Precision (Précision)": precision_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred),
    }

    print("\n--- Métriques sur l'Ensemble de Test ---")
    for name, value in metrics.items():
        print(f"   * {name}: {value:.4f}")
        
    return metrics

# ==============================================================================
# 6. Exécution Principale
# ==============================================================================

def main():
    
    # 1. Chargement des données
    X, y = load_data(FILE_PATH)
    if X is None:
        return

    # --- Nettoyage des lignes avec dates non valides (NaT) après conversion ---
    # Ces lignes ne peuvent pas être utilisées pour le split temporel.
    X_clean = X.dropna(subset=['dateCreationUniteLegale'])
    if X_clean.empty:
        print("ERREUR FATALE: Aucune date de création valide trouvée après nettoyage.")
        return
        
    y_clean = y[X_clean.index]
    print(f"\nNettoyage des {len(X) - len(X_clean)} lignes avec dates non valides ou manquantes.")
    X = X_clean
    y = y_clean
    # -------------------------------------------------------------------------

    # 2. Split Temporel
    X_train, X_test, y_train, y_test = temporal_split(X, y, SPLIT_DATE)
    if X_train is None:
        return # Arrêt si le split a échoué

    # 3. Pré-traitement
    preprocessor = build_preprocessor(X_train)
    
    # --- BASELINE : RÉGRESSION LOGISTIQUE ---
    print("\n\n#####################################################")
    print("##             EXPERIENCE 1: BASELINE              ##")
    print("#####################################################")
    
    # 4. Construction et Entraînement du Pipeline Baseline
    baseline_pipeline = build_model_pipeline(preprocessor, model_name='baseline')
    print("\n[DÉBUT] Entraînement de la Baseline (LogReg). Peut prendre du temps...")
    baseline_pipeline.fit(X_train, y_train)
    print("[FIN] Entraînement de la Baseline.")
    
    # 5. Évaluation et CV
    run_cross_validation(baseline_pipeline, X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_test, y_test)
    
    # --- ITÉRATION 1 : RANDOM FOREST ---
    print("\n\n#####################################################")
    print("##         EXPERIENCE 2: RANDOM FOREST (ITÉRATION) ##")
    print("#####################################################")
    
    # 4. Construction et Entraînement du Pipeline Random Forest
    rf_pipeline = build_model_pipeline(preprocessor, model_name='iteration_rf')
    print("\n[DÉBUT] Entraînement du Random Forest. Plus long que la LogReg...")
    rf_pipeline.fit(X_train, y_train)
    print("[FIN] Entraînement du Random Forest.")
    
    # 5. Évaluation et CV
    run_cross_validation(rf_pipeline, X_train, y_train)
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)
    
    # Comparaison des résultats
    print("\n\n--- COMPARAISON DES EXPÉRIENCES FINALES ---")
    print(f"Baseline (LogReg) ROC AUC: {baseline_metrics['ROC AUC']:.4f}")
    print(f"Itération 1 (RF) ROC AUC: {rf_metrics['ROC AUC']:.4f}")


if __name__ == '__main__':
    main()