import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

# --- 0. DÉFINITION GLOBALE DES FONCTIONS ROBUSTES ---

# Chemin d'accès au fichier ML (ajuster si nécessaire)
FILE_PATH_ML = "Data/processed/sirene_bilan_ML_prets.parquet" 
cible_col = "cible_HN_RésultatNet_T_plus_1"

# Fonction de transformation robuste de la cible (ARCSINH)
def arcsinh_transform_safe(y):
    # Transformation recommandée pour les données financières (gains et pertes)
    # Ajout d'epsilon pour la robustesse près de zéro
    return np.arcsinh(y + np.finfo(float).eps)

# Fonction d'inverse transformation
def inv_arcsinh_transform_safe(y_pred_arcsinh):
    # Inverse de arcsinh
    return np.sinh(y_pred_arcsinh) - np.finfo(float).eps

# Scoreurs pour la Cross-Validation
def root_mean_squared_error(y_true, y_pred):
    # Fonction RMSE avec conversion pour éviter l'overflow
    return np.sqrt(mean_squared_error(y_true.astype(np.float64), y_pred.astype(np.float64)))

scorer_mae = make_scorer(mean_absolute_error, greater_is_better=False)
scorer_rmse = make_scorer(root_mean_squared_error, greater_is_better=False)

# Features finales retenues pour le modèle performant
FEATURES_FINALES = [
    'ratio_rentabilite_nette', 'ratio_endettement', 'ratio_marge_brute', 
    'HN_RésultatNet', 'FA_ChiffreAffairesVentes', 
    'delta_ResultatNet_1an', 'delta_CA_1an', 'ResultatNet_T_moins_1', 'CA_T_moins_1'
]


# --- CLASSE PRINCIPALE D'ENTRAÎNEMENT ---
class FinalModelTrainer:
    
    def __init__(self):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
    def load_and_prepare_data(self):
        """Charge, nettoie et transforme les données."""
        print("Chargement et préparation des données...")
        try:
            df_ml = pl.read_parquet(FILE_PATH_ML)
        except Exception:
            raise RuntimeError("Erreur de chargement du fichier Parquet.")

        df_ml_pd = df_ml.to_pandas()
        
        # Application de la transformation ARCSINH à la Cible (Y)
        Y_full_arcsinh = arcsinh_transform_safe(df_ml_pd[cible_col].astype(np.float64))

        # Préparation des Features X
        X_full = df_ml_pd[FEATURES_FINALES].fillna(0).astype(np.float64) 
        
        print(f"Jeu de données prêt : {X_full.shape[0]} observations.")
        return X_full, Y_full_arcsinh

    def build_and_evaluate_pipeline(self, X, Y):
        """Construit le pipeline (Gradient Boosting) et évalue par CV."""
        print("\nConstruction du pipeline Gradient Boosting (Modèle Monstre)...")
        
        # Pre-processing (StandardScaler)
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), FEATURES_FINALES)],
            remainder='passthrough'
        )

        # Modèle Gradient Boosting (similaire à XGBoost mais utilise scikit-learn)
        # Ces paramètres sont choisis pour la robustesse et la performance.
        model_gbr = GradientBoostingRegressor(
            n_estimators=500,  # Nombre d'arbres élevé
            learning_rate=0.05,
            max_depth=5,
            subsample=0.7,
            random_state=42
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_gbr)
        ])
        
        # --- Évaluation par Cross-Validation (CV) ---
        print("Évaluation par 5-Fold Cross-Validation...")
        start_time = time.time()
        
        rmse_scores = cross_val_score(pipeline, X, Y, scoring=scorer_rmse, cv=self.kf, n_jobs=-1)
        mae_scores = cross_val_score(pipeline, X, Y, scoring=scorer_mae, cv=self.kf, n_jobs=-1)
        
        training_time = time.time() - start_time
        
        # Calcul des moyennes (multiplié par -1 car les scoreurs sont négatifs)
        rmse_cv_mean = np.mean(rmse_scores) * -1
        mae_cv_mean = np.mean(mae_scores) * -1
        
        return pipeline, mae_cv_mean, rmse_cv_mean, training_time

    def inverse_transform_and_evaluate(self, pipeline, X_full, Y_full_arcsinh, final_mae_cv, final_rmse_cv, training_time):
        """Inverse la transformation et affiche les résultats finaux."""
        
        # Entraîner le modèle final sur toutes les données avant la prédiction
        pipeline.fit(X_full, Y_full_arcsinh)

        # Prédiction (Arcsih-transformée)
        Y_pred_arcsinh = pipeline.predict(X_full)

        # Inverse Transformation : Retour aux unités monétaires originales
        Y_pred_final_unscaled = inv_arcsinh_transform_safe(Y_pred_arcsinh)
        Y_true_unscaled = inv_arcsinh_transform_safe(Y_full_arcsinh)

        # Affichage des métriques sur les valeurs RÉELLES
        final_mae_unscaled = mean_absolute_error(Y_true_unscaled, Y_pred_final_unscaled)
        final_rmse_unscaled = root_mean_squared_error(Y_true_unscaled, Y_pred_final_unscaled)

        print("\n=============================================")
        print("🏆 MODÈLE MONSTRE FINAL (Gradient Boosting) 🏆")
        print("=============================================")
        print(f"TEMPS TOTAL D'ENTRAÎNEMENT CV : {training_time:.2f} secondes")
        print(f"Features utilisées : Top {len(FEATURES_FINALES)} (Validées par EDA)")
        print("-" * 45)
        print(f"  > MAE (Erreur Absolue Moyenne, Unscaled) : {final_mae_unscaled:,.2f}")
        print(f"  > RMSE (Racine de l'Erreur Quadratique, Unscaled) : {final_rmse_unscaled:,.2f}")
        print(f"  > MAE (Moyenne CV, Interne) : {final_mae_cv:,.2f} (Confirmé par CV)")
        print("=============================================")


# --- EXÉCUTION DU PIPELINE COMPLET ---
if __name__ == "__main__":
    trainer = FinalModelTrainer()
    
    # 1. Chargement et Transformation
    X_full, Y_full_arcsinh = trainer.load_and_prepare_data()
    
    # 2. Construction et Évaluation du Pipeline
    pipeline_final, mae_cv, rmse_cv, training_time = trainer.build_and_evaluate_pipeline(X_full, Y_full_arcsinh)
    
    # 3. Affichage des résultats finaux (avec inverse transformation)
    trainer.inverse_transform_and_evaluate(pipeline_final, X_full, Y_full_arcsinh, mae_cv, rmse_cv, training_time)