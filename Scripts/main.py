import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Importation critique pour gérer les NaN
# Modèles
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor 
# Métriques
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
import warnings
import os 
import shutil

# Configuration pour une exécution plus propre
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURATION ET PARAMÈTRES ---
TARGET = 'cible_ResultatNet_T_plus_1'
DATA_PATH = 'data/processed/sirene_infos_FINAL.parquet'
RANDOM_STATE = 42
MODEL_ARTIFACT_PATH = "best_model_artifact" 

# Colonnes financières (à transformer log et à imputer avec 0)
LOG_TRANSFORM_COLS = [
    'CJCK_TotalActifBrut', 'HN_RésultatNet', 'DA_TresorerieActive', 
    'DL_DettesCourtTerme', TARGET
]
# Autres colonnes numériques
NUMERICAL_FEATURES_SCALED = [
    'ratio_endettement', 'ratio_tresorerie', 'anciennete_entreprise'
]
NUMERICAL_FEATURES_ALL = LOG_TRANSFORM_COLS[:-1] + NUMERICAL_FEATURES_SCALED

CATEGORICAL_FEATURES = [
    'trancheEffectifsUniteLegale', 'activitePrincipaleUniteLegale', 
    'economieSocialeSolidaireUniteLegale', 'departement', 
    'caractereEmployeurSiege'
]

# --- 2. FONCTIONS DE TRANSFORMATION ---

def signed_log_transform(x):
    """Applique une transformation log(1 + |x|) en préservant le signe."""
    return np.sign(x) * np.log1p(np.abs(x))

def inverse_signed_log_transform(y_pred):
    """Inverse la transformation pour ramener les prédictions à l'échelle monétaire réelle."""
    return np.sign(y_pred) * (np.exp(np.abs(y_pred)) - 1)

# --- 3. FONCTIONS DE PIPELINE ---

def load_and_transform_data(file_path):
    """Charge, consolide, gère les NaNs (imputation), puis applique les transformations log."""
    print(f"Chargement des données depuis : {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except FileNotFoundError:
        return None

    # Consolidation
    data = data.sort_values(by=['siren', 'date_cloture_exercice'], ascending=[True, False])
    data_consolidated = data.drop_duplicates(
        subset=['siren', 'date_cloture_exercice'], keep='first'
    ).reset_index(drop=True)
    
    # 1. Gestion des valeurs nulles AVANT la transformation (features)
    data_consolidated[NUMERICAL_FEATURES_ALL] = data_consolidated[NUMERICAL_FEATURES_ALL].fillna(0)
    data_consolidated[CATEGORICAL_FEATURES] = data_consolidated[CATEGORICAL_FEATURES].fillna('MISSING')

    # 2. Application de la Transformation Log-Signée
    for col in LOG_TRANSFORM_COLS:
        data_consolidated[col] = signed_log_transform(data_consolidated[col])
        
    # --- NETTOYAGE CRITIQUE DE LA CIBLE POST-TRANSFORMATION ---
    # La transformation log a pu générer des NaN/Inf pour des valeurs extrêmes. Il faut les supprimer.
    initial_rows = data_consolidated.shape[0]
    data_consolidated = data_consolidated[
        np.isfinite(data_consolidated[TARGET])
    ]
    
    rows_dropped = initial_rows - data_consolidated.shape[0]
    if rows_dropped > 0:
        print(f"ATTENTION : {rows_dropped} lignes ({rows_dropped/initial_rows*100:.2f}%) supprimées car la transformation log a généré NaN/Inf dans la cible.")
    
    print(f"Données consolidées et nettoyées. Taille finale : {data_consolidated.shape}")
    return data_consolidated

def get_preprocessor(numerical_features, categorical_features):
    """Crée le ColumnTransformer avec Imputer pour garantir l'absence de NaN."""
    
    # Correction : Imputer pour garantir l'absence de NaN après toutes les étapes
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Correction : Ignorer les colonnes non spécifiées.
    )
    return preprocessor

def evaluate_model(y_true_transformed, y_pred_transformed, model_name):
    """Calcule les métriques clés, inversant la transformation pour RMSE/MAE."""
    
    r2_transformed = r2_score(y_true_transformed, y_pred_transformed)
    
    # Inversion de la transformation pour les métriques métier
    y_true_real = inverse_signed_log_transform(y_true_transformed)
    y_pred_real = inverse_signed_log_transform(y_pred_transformed)
    
    rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real)) 
    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    
    print(f"\n--- Métriques pour {model_name} ---")
    print(f"R2 Score (Transformé): {r2_transformed:.4f} (Métriques d'ajustement)")
    print(f"RMSE (Réel): {rmse_real:.2f} (Échelle monétaire)")
    print(f"MAE (Réel): {mae_real:.2f} (Échelle monétaire)")
    
    return {"rmse_real": rmse_real, "mae_real": mae_real, "r2_transformed": r2_transformed}

def plot_feature_importance(pipeline, X_train, model_name):
    """Affiche l'importance des features après l'entraînement d'un modèle basé sur les arbres."""
    
    if not hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        return

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
    
    # Sauvegarder le graphique via MLflow
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f"Top 20 Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name}.png")
    mlflow.log_artifact(f"feature_importance_{model_name}.png")
    plt.close()
    

def run_experiment(model, model_name, X_train, y_train, X_test, y_test, preprocessor, cv_folds=5):
    """Entraîne, évalue, loggue l'expérience et analyse l'importance."""
    
    with mlflow.start_run(run_name=model_name) as run:
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        mlflow.log_params(full_pipeline.named_steps['regressor'].get_params())
        
        # Cross-Validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
        mean_cv_r2 = np.mean(cv_scores)
        print(f"\nExécution de la Cross-Validation ({cv_folds} folds)...")
        print(f"R2 Cross-Validation Moyenne (Transformé): {mean_cv_r2:.4f}")
        mlflow.log_metric("cv_mean_r2", mean_cv_r2)
        
        # Entraînement final
        full_pipeline.fit(X_train, y_train)
        
        # Prédiction et Évaluation
        y_pred = full_pipeline.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred, model_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Analyse de l'importance des features
        plot_feature_importance(full_pipeline, X_train, model_name)
        
        mlflow.sklearn.log_model(full_pipeline, "model")
        
        return test_metrics, full_pipeline

# --- 4. PIPELINE PRINCIPAL ---
if __name__ == "__main__":
    
    # 0. Gestion de l'erreur MLflow (suppression du dossier pour réexécution)
    if os.path.exists(MODEL_ARTIFACT_PATH):
        shutil.rmtree(MODEL_ARTIFACT_PATH)

    # 1. Chargement, Consolidation et Transformation
    data = load_and_transform_data(DATA_PATH)
    if data is None:
        exit()

    # 2. Séparation Train-Test
    COLS_TO_EXCLUDE = [TARGET, 'siren', 'date_cloture_exercice', 'AnneeClotureExercice']
    X = data.drop(columns=COLS_TO_EXCLUDE)
    y = data[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"Taille du jeu d'entraînement: {X_train.shape[0]} lignes")
    print(f"Taille du jeu de test: {X_test.shape[0]} lignes")

    # 3. Préparation du ColumnTransformer
    preprocessor = get_preprocessor(NUMERICAL_FEATURES_ALL, CATEGORICAL_FEATURES)
    
    # 4. Définition des expérimentations
    
    experiments = [
        {'name': 'Baseline_Ridge_LogScale', 
         'model': Ridge(alpha=1.0, random_state=RANDOM_STATE)}, 
        
        {'name': 'Iteration_1_XGBoost_Analysis', 
         'model': XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)},
        
        {'name': 'Iteration_2_LightGBM_Tuned', 
         'model': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, reg_lambda=0.5, random_state=RANDOM_STATE, n_jobs=-1)},
    ]

    # 5. Exécution des expérimentations et suivi du meilleur modèle
    mlflow.set_experiment("Prediction_ResultatNet_Financier_V3_LogTransform_FS")
    
    best_r2_transformed = -np.inf
    best_model = None
    
    print("\n--- Démarrage des Expérimentations (Tracking avec MLflow) ---")
    for exp in experiments:
        metrics, pipeline = run_experiment(
            model=exp['model'],
            model_name=exp['name'],
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test,
            preprocessor=preprocessor
        )
        
        if metrics['r2_transformed'] > best_r2_transformed:
            best_r2_transformed = metrics['r2_transformed']
            best_model = pipeline
            
    # 6. Sauvegarde du meilleur modèle
    if best_model:
        print(f"\nMeilleur modèle (basé sur le R2 transformé) sauvegardé: {best_model.named_steps['regressor'].__class__.__name__}")
        mlflow.sklearn.save_model(best_model, MODEL_ARTIFACT_PATH)