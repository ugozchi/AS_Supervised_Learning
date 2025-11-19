import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

# Configuration pour une exécution plus propre
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURATION ET PARAMÈTRES ---
TARGET = 'cible_ResultatNet_T_plus_1'
DATA_PATH = 'data/processed/sirene_infos_FINAL.parquet'
RANDOM_STATE = 42

# Colonnes fortement asymétriques à transformer (inclut la cible)
LOG_TRANSFORM_COLS = [
    'CJCK_TotalActifBrut', 'HN_RésultatNet', 'DA_TresorerieActive', 
    'DL_DettesCourtTerme', TARGET
]
# Ratios et Ancienneté
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
    """Charge, consolide, gère les NaNs, puis applique les transformations log."""
    print(f"Chargement des données depuis : {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"ERREUR: Fichier {file_path} non trouvé. Arrêt.")
        return None

    # Consolidation : garder la ligne la plus récente pour chaque paire (siren, date_cloture_exercice)
    data = data.sort_values(by=['siren', 'date_cloture_exercice'], ascending=[True, False])
    data_consolidated = data.drop_duplicates(
        subset=['siren', 'date_cloture_exercice'], keep='first'
    ).reset_index(drop=True)
    
    print(f"Données consolidées. Taille : {data_consolidated.shape}")
    
    # 1. Gestion des valeurs nulles AVANT la transformation (Corrige l'erreur NaN)
    ALL_NUMERICAL_TO_IMPUTE = LOG_TRANSFORM_COLS[:-1] + NUMERICAL_FEATURES_SCALED
    data_consolidated[ALL_NUMERICAL_TO_IMPUTE] = data_consolidated[ALL_NUMERICAL_TO_IMPUTE].fillna(0)
    data_consolidated[CATEGORICAL_FEATURES] = data_consolidated[CATEGORICAL_FEATURES].fillna('MISSING')

    # Suppression des lignes sans cible (après remplissage des features)
    data_consolidated.dropna(subset=[TARGET], inplace=True)

    # 2. Application de la Transformation Log-Signée
    for col in LOG_TRANSFORM_COLS:
        data_consolidated[col] = signed_log_transform(data_consolidated[col])
        
    print("Transformation Log-Signée appliquée aux features financières et à la cible.")
    return data_consolidated

def get_preprocessor(numerical_features, categorical_features):
    """Crée le ColumnTransformer (StandardScaler pour num, OneHotEncoding pour cat)."""
    numerical_transformer = Pipeline(steps=[
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
        remainder='passthrough'
    )
    return preprocessor

def evaluate_model(y_true_transformed, y_pred_transformed, model_name):
    """Calcule les métriques clés, inversant la transformation pour RMSE/MAE."""
    
    # Métriques sur l'échelle transformée
    r2_transformed = r2_score(y_true_transformed, y_pred_transformed)
    
    # Inversion de la transformation pour les métriques métier
    y_true_real = inverse_signed_log_transform(y_true_transformed)
    y_pred_real = inverse_signed_log_transform(y_pred_transformed)
    
    # Métriques sur l'échelle réelle
    rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real)) 
    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    
    print(f"\n--- Métriques pour {model_name} ---")
    print(f"R2 Score (Transformé): {r2_transformed:.4f}")
    print(f"RMSE (Réel): {rmse_real:.2f}")
    print(f"MAE (Réel): {mae_real:.2f}")
    
    return {"rmse_real": rmse_real, "mae_real": mae_real, "r2_transformed": r2_transformed}

def plot_feature_importance(pipeline, X_train, model_name):
    """Affiche l'importance des features après l'entraînement d'un modèle basé sur les arbres."""
    
    if not hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
        print(f"Skipping feature importance for {model_name}: Model does not have 'feature_importances_'.")
        return

    # 1. Obtenir les noms des features après l'encodage OHE
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # 2. Récupérer l'importance
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    # 3. Créer un DataFrame et trier
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
    
    # 4. Enregistrer l'importance dans MLflow
    mlflow.log_dict({"top_20_features": feature_importance_df.to_dict()}, "feature_importance.json")
    
    # 5. Visualisation
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f"Top 20 Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name}.png")
    mlflow.log_artifact(f"feature_importance_{model_name}.png")
    plt.close() # Fermer la figure pour libérer la mémoire
    

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
        mlflow.log_metric("cv_mean_r2", mean_cv_r2)
        
        # Entraînement final
        full_pipeline.fit(X_train, y_train)
        
        # Prédiction et Évaluation
        y_pred = full_pipeline.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred, model_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Analyse de l'importance des features (pour modèles arborescents)
        plot_feature_importance(full_pipeline, X_train, model_name)
        
        mlflow.sklearn.log_model(full_pipeline, "model")
        
        return test_metrics, full_pipeline

# --- 4. STRATÉGIE D'ITÉRATION POUR LA FEATURE SELECTION ---

def feature_selection_strategy(best_model_run_id):
    """
    Simule la stratégie d'itération basée sur l'importance des features 
    pour réduire l'Overfitting.
    """
    print("\n--- Analyse de la Feature Importance pour la réduction de l'Overfitting ---")
    
    # 1. Récupérer l'artefact d'importance du meilleur run
    # Normalement, on chargerait les données du JSON/DF du meilleur run ici.
    
    # 2. Identifier les features à faible importance (ex: les 80% les moins importants)
    # L'objectif est de réduire la dimensionnalité, surtout après l'OHE.
    
    # 3. Créer une nouvelle expérience (Itération 4) avec les features sélectionnées.
    
    print("Pour les prochaines itérations, focalisez-vous sur les 10-20 features les plus importantes (issues du plot) et relancez le modèle (XGBoost/LGBM) avec ces features uniquement.")
    # Ceci nécessite de modifier la liste des NUMERICAL_FEATURES_ALL et CATEGORICAL_FEATURES
    # dans le script pour l'Itération 4.

# --- 5. PIPELINE PRINCIPAL ---
if __name__ == "__main__":
    
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

    # 3. Préparation du ColumnTransformer
    preprocessor = get_preprocessor(NUMERICAL_FEATURES_ALL, CATEGORICAL_FEATURES)
    
    # 4. Définition des expérimentations
    
    experiments = [
        {'name': 'Baseline_Ridge_LogScale', 
         'model': Ridge(alpha=1.0, random_state=RANDOM_STATE)}, 
        
        {'name': 'Iteration_1_XGBoost_Analysis', 
         'model': XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)},
        
        {'name': 'Iteration_2_LightGBM_Analysis', 
         'model': LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)},
    ]

    # 5. Exécution des expérimentations
    mlflow.set_experiment("Prediction_ResultatNet_Financier_V3_LogTransform_FS")
    
    best_r2_transformed = -np.inf
    best_model_run_id = None
    
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
            best_model_run_id = mlflow.active_run().info.run_id
            
    # 6. Stratégie Post-Entraînement
    feature_selection_strategy(best_model_run_id)

    # Note sur l'erreur MLflow : Supprimer le dossier 'best_model_artifact' avant de relancer.
    # Pour ne plus avoir l'erreur : utilisez mlflow.register_model() au lieu de save_model()
    # si vous souhaitez enregistrer le meilleur modèle.