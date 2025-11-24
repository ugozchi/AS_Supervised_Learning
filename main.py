import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, roc_auc_score
import category_encoders as ce  
import pandas as pd  
import warnings
import os
import polars as pl
import sys
from typing import Tuple, List

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================================================
## 0. CONFIGURATION & FONCTIONS UTILITAIRES
# ==================================================

DATA_PATH = 'Data/processed/sirene_final2.parquet'
TARGET_RN_NPLUS1 = 'Y_RN' 
RANDOM_SEED = 42

# --- DÉFINITIONS DES COLONNES ---
TARGET_ENCODING_COLS = ['departement', 'secteur_NAF_2chiffres']
COLUMNS_TO_REMOVE_BRUT = [
    'siren', 'dateCreationUniteLegale', 'AnneeClotureExercice', 
    'activitePrincipaleUniteLegale', 
    'HN_RésultatNet', 'FR_ResultatExceptionnel', 
    'DL_DettesCourtTerme', 'CJCK_TotalActifBrut',
    'anciennete',
    'categorieJuridiqueUniteLegale', 'trancheEffectifsUniteLegale', 
    'categorieEntreprise', 'trancheEffectifsSiege', 'caractereEmployeurSiege',
]

# ... (Fonctions load_data, safe_divide, safe_log1p inchangées)

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERREUR FATALE: Fichier non trouvé: {file_path}")
    
    try:
        df_pl = pl.read_parquet(file_path)
        df = df_pl.to_pandas()
        print(f"DEBUG: Chargement réussi. Lignes brutes: {len(df)}")
        return df
    except Exception as e:
        raise RuntimeError(f"ERREUR: Impossible de lire le fichier Parquet. Détail: {e}")

def safe_divide(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator_safe = denominator.copy() 
    else:
        denominator_safe = pd.Series(denominator) 
        
    denominator_safe[denominator_safe == 0] = 1e-8 
    ratio = numerator / denominator_safe
    
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

def safe_log1p(series):
    return np.sign(series) * np.log1p(np.abs(series))


# ==================================================
## 1. DATA PREP & FEATURE ENGINEERING
# ==================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
    # --- CRÉATION DE FEATURES NUMÉRIQUES ---
    df['ratio_liquidite'] = safe_divide(df['DA_TresorerieActive'], df['DL_DettesCourtTerme'])
    df['ratio_stabilite_inv'] = safe_divide(1, df['anciennete'])
    df['proxy_actif_taux'] = safe_divide(df['DA_TresorerieActive'] - df['DL_DettesCourtTerme'], df['anciennete'])
    df['HN_RésultatNet_log'] = safe_log1p(df['HN_RésultatNet'])
    df['FR_ResultatExceptionnel_log'] = safe_log1p(df['FR_ResultatExceptionnel'])
    df['flag_exceptionnel'] = (df['FR_ResultatExceptionnel'] != 0).astype(int)

    df['secteur_NAF_2chiffres'] = df['activitePrincipaleUniteLegale'].astype(str).str[:2]
    
    df['DL_DettesCourtTerme_log'] = safe_log1p(df['DL_DettesCourtTerme'])
    df['CJCK_TotalActifBrut_log'] = safe_log1p(df['CJCK_TotalActifBrut'])
    df['ratio_dette_ct_vs_actif'] = safe_divide(df['DL_DettesCourtTerme'], df['CJCK_TotalActifBrut'])
    df['ratio_tresorerie_vs_dette_ct'] = safe_divide(df['DA_TresorerieActive'], df['DL_DettesCourtTerme'])
    
    # --- SUPPRESSION MAXIMALE DES COLONNES BRUTES ---
    df = df.drop(columns=COLUMNS_TO_REMOVE_BRUT, errors='ignore')
            
    # --- Dérivation des Cibles Hybrides ---
    df['target_is_profit'] = (df[TARGET_RN_NPLUS1] > 0).astype(int)
    df['target_magnitude_log'] = np.log1p(np.abs(df[TARGET_RN_NPLUS1]))

    # --- Nettoyage Final des NaN/Inf ---
    df = df.replace([np.inf, -np.inf], np.nan) 
    df = df.dropna(subset=[TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log'])
    print(f"DEBUG: Lignes après Feature Engineering & Nettoyage: {len(df)}")
    
    return df

# ==================================================
## 2. DÉFINITION X, Y ET TRAIN/TEST SPLIT
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    """Sépare les données et effectue le split Train/Test (Index préservé pour le look-up final)."""
    
    # FLAG_VERIF: La liste des colonnes à dropper est vide car le nettoyage brutal a déjà eu lieu.
    COLUMNS_TO_DROP_FINAL = [
        TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 
    ]
    
    X = df.drop(columns=COLUMNS_TO_DROP_FINAL, errors='ignore')
    
    # Assurer que les colonnes Target Encoding sont 'object'
    for col in TARGET_ENCODING_COLS:
        if col in X.columns:
            X[col] = X[col].astype('object')
            
    Y_reg_log = df['target_magnitude_log']
    Y_cls = df['target_is_profit']
    
    # Pas de réinitialisation d'index ici pour X et Y. 
    # Le split va conserver les index désordonnés.
    X_train, X_test, Y_reg_log_train, Y_reg_log_test, Y_cls_train, Y_cls_test = train_test_split(
        X, Y_reg_log, Y_cls, test_size=0.2, random_state=RANDOM_SEED, stratify=Y_cls
    )
    
    print(f"Shapes (Train: {X_train.shape}, Test: {X_test.shape})")
    
    return X_train, X_test, Y_reg_log_train, Y_reg_log_test, Y_cls_train, Y_cls_test


# ==================================================
## 3. ÉTAPE 1 : CLASSIFICATION DU SIGNE (PROFIT ou PERTE)
# ==================================================

def train_and_evaluate_cls(X_train, X_test, Y_train, Y_test) -> Tuple[xgb.XGBClassifier, pd.DataFrame]:
    """Entraîne et évalue le modèle de Classification Binaire (Signe)."""

    cls_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS)
    
    X_train_encoded = cls_encoder.fit_transform(X_train, Y_train)
    X_test_encoded = cls_encoder.transform(X_test)

    print("\n[MODELE CLS] Entraînement pour prédire le SIGNE (Profit/Perte)...")
    cls_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        n_estimators=700, learning_rate=0.03, max_depth=7,
        random_state=RANDOM_SEED, n_jobs=-1,
        subsample=0.8, colsample_bytree=0.8
    )
    cls_model.fit(X_train_encoded, Y_train)

    cls_pred = cls_model.predict(X_test_encoded)
    cls_accuracy = accuracy_score(Y_test, cls_pred)

    print("\n--- PERFORMANCE CLASSIFICATION (Signe) ---")
    print(f"Accuracy : {cls_accuracy:.4f}")
    
    return cls_model, X_test_encoded 

# ==================================================
## 4. ÉTAPE 2 : RÉGRESSION DE LA MAGNITUDE (LOG)
# ==================================================

def train_and_evaluate_reg(X_train, Y_train):
    """Effectue la Cross-Validation (CV) et entraîne le modèle final de Régression (Magnitude Log)."""
    
    print("\n[MODELE REG] Cross-Validation (CV=5) avec TARGET ENCODING BOOSTÉ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores_cv = []

    base_regressor = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1500, max_depth=8, learning_rate=0.03, 
        subsample=0.7, colsample_bytree=0.7, 
        random_state=RANDOM_SEED, n_jobs=-1
    )

    # Boucle de Cross-Validation SUR X_train / Y_train
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)): 
        
        # Slicing par position (.iloc) et copie (la seule façon de garantir l'alignement dans les folds)
        X_train_fold = X_train.iloc[train_index].copy() 
        X_val_fold = X_train.iloc[val_index].copy()
        Y_reg_log_train_fold = Y_train.iloc[train_index].copy()
        Y_reg_log_val_fold = Y_train.iloc[val_index].copy() # Cible de validation
        
        reg_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS)
        X_train_encoded = reg_encoder.fit_transform(X_train_fold, Y_reg_log_train_fold)
        X_val_encoded = reg_encoder.transform(X_val_fold)
        
        fold_reg_model = base_regressor
        fold_reg_model.fit(X_train_encoded, Y_reg_log_train_fold) 
        
        pred_log_mag = fold_reg_model.predict(X_val_encoded)
        r2_scores_cv.append(r2_score(Y_reg_log_val_fold, pred_log_mag))
        
        print(f"Fold {fold+1}: R2={r2_scores_cv[-1]:.4f}")

    print("\n--- PERFORMANCE MOYENNE CROSS-VALIDATION (Magnitude Log) ---")
    print(f"R² (CV)  : {np.mean(r2_scores_cv):.4f}")

    # Entraînement du modèle final sur l'ensemble TRAIN complet 
    final_reg_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS) 
    X_train_final = X_train.copy()
    X_train_final_encoded = final_reg_encoder.fit_transform(X_train_final, Y_train)
    reg_model_final = base_regressor
    reg_model_final.fit(X_train_final_encoded, Y_train)
    
    return reg_model_final, final_reg_encoder

# ==================================================
## 5. PRÉDICTION FINALE & ÉVALUATION GLOBALE
# ==================================================

def evaluate_global(cls_model, reg_model, reg_encoder, X_test, Y_cls_test, df_processed):
    """Combine les prédictions et évalue le Résultat Net final."""

    # --- Étape 1: Préparation des DataFrames Test ---
    # FLAG_VERIF: Utiliser .copy() pour s'assurer que l'encodage ne modifie pas l'objet X_test original
    X_test_reg = X_test.copy()
    X_test_reg_encoded = reg_encoder.transform(X_test_reg)
    
    predictions_magnitude_log = reg_model.predict(X_test_reg_encoded)
    predictions_magnitude = np.expm1(predictions_magnitude_log)
    predictions_magnitude[predictions_magnitude < 0] = 0

    # --- Étape 2: Prédiction du Signe ---
    cls_encoder_final = ce.TargetEncoder(cols=TARGET_ENCODING_COLS) 
    
    # Fit l'encoder sur l'ensemble complet (après Feature Engineering)
    X_fit_cls = df_processed.drop(columns=[TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log'], errors='ignore')
    Y_fit_cls = df_processed['target_is_profit']
    cls_encoder_final.fit(X_fit_cls, Y_fit_cls)
    
    X_test_cls = X_test.copy()
    X_test_cls_encoded = cls_encoder_final.transform(X_test_cls)
    
    predictions_signe = np.where(cls_model.predict(X_test_cls_encoded) == 1, 1, -1)
    
    # --- Étape 3: Combinaison et Évaluation ---
    final_predictions = predictions_signe * predictions_magnitude

    # CORRECTION CRITIQUE (Résout la KeyError finale):
    # Nous utilisons l'index du Y_cls_test (qui a le bon index original) pour faire le lookup dans df_processed.
    y_test_original = df_processed.loc[Y_cls_test.index, TARGET_RN_NPLUS1] 
    
    final_mae = mean_absolute_error(y_test_original, final_predictions)
    final_r2 = r2_score(y_test_original, final_predictions)

    print("\n--- PERFORMANCE GLOBALE FINALE (Test Set) ---")
    print(f"MAE : {final_mae:,.2f} € ")
    print(f"R²  : {final_r2:.4f} ")


# ==================================================
## 6. EXÉCUTION DU PIPELINE
# ==================================================

def main():
    print("--- Démarrage du Pipeline Hybride C/R (Prédiction de Y_RN) ---")
    
    try:
        df_raw = load_data(DATA_PATH)
        df_processed = feature_engineering(df_raw)
        
        # Les jeux X_train/X_test ont désormais les INDEX ORIGINAUX du df_processed
        X_train, X_test, Y_reg_log_train, Y_reg_log_test, Y_cls_train, Y_cls_test = split_data(df_processed)
        
        cls_model, X_test_cls_encoded = train_and_evaluate_cls(X_train, X_test, Y_cls_train, Y_cls_test)

        # La CV utilise X_train / Y_train (qui ont les index alignés)
        reg_model, reg_encoder = train_and_evaluate_reg(X_train, Y_reg_log_train) 
        
        # L'évaluation utilise les INDEX ORIGINAUX de Y_cls_test pour le lookup dans df_processed
        evaluate_global(cls_model, reg_model, reg_encoder, X_test, Y_cls_test, df_processed)
        
        print("\n--- Pipeline Hybride C/R Terminé avec Succès ---")

    except Exception as e:
        print(f"\nFATAL ERROR: Le script s'est arrêté en raison d'une erreur critique: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()