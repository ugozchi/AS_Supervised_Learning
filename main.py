import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import category_encoders as ce  
import pandas as pd  
import warnings
import os
import polars as pl
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================================================
## 0. CONFIGURATION & FONCTIONS UTILITAIRES
# ==================================================

DATA_PATH = 'Data/processed/sirene_final2.parquet'
TARGET_RN_NPLUS1 = 'Y_RN' 
RANDOM_SEED = 42

TARGET_ENCODING_COLS = ['departement', 'secteur_NAF_2chiffres']

FINAL_FEATURE_WHITELIST = [
    'ratio_rentabilite_nette', 'ratio_endettement', 'ratio_tresorerie', 
    'ratio_resultat_financier', 'ratio_resultat_exceptionnel', 
    'ratio_liquidite', 'ratio_stabilite_inv', 'proxy_actif_taux',
    'HN_RésultatNet_log', 'FR_ResultatExceptionnel_log', 
    'DL_DettesCourtTerme_log', 'CJCK_TotalActifBrut_log',
    'ratio_dette_ct_vs_actif', 'ratio_tresorerie_vs_dette_ct', 
    'flag_exceptionnel', 
    'taux_croissance_RN_N-1', 'taux_croissance_RN_N-2', 
    'departement', 'secteur_NAF_2chiffres', 
]

COLS_TO_WINSORIZE = [
    TARGET_RN_NPLUS1, 'HN_RésultatNet', 'CJCK_TotalActifBrut', 
    'DL_DettesCourtTerme', 'FR_ResultatExceptionnel'
]
WINSOR_LOW = 0.025
WINSOR_HIGH = 0.975


def load_data(file_path: str) -> pd.DataFrame:
    """Charge le dataset Parquet."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERREUR: Fichier non trouvé: {file_path}")
    
    df_pl = pl.read_parquet(file_path)
    return df_pl.to_pandas()

def safe_divide(numerator, denominator):
    """Division sécurisée."""
    if isinstance(denominator, pd.Series):
        denominator_safe = denominator.copy() 
    else:
        denominator_safe = pd.Series(denominator) 
        
    denominator_safe[denominator_safe == 0] = 1e-8 
    ratio = numerator / denominator_safe
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

def safe_log1p(series):
    """Log avec gestion du signe."""
    return np.sign(series) * np.log1p(np.abs(series))

# ==================================================
## 1. FEATURE ENGINEERING (CORRIGÉ)
# ==================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering SANS data leakage."""
    
    print("\n🔧 Application de la Winsorisation...")
    for col in COLS_TO_WINSORIZE:
        if col in df.columns:
            lower_bound = df[col].quantile(WINSOR_LOW)
            upper_bound = df[col].quantile(WINSOR_HIGH)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # ✅ CORRECTION: Pas de shift() global - utilisation directe des variations
    df['taux_croissance_RN_N-1'] = safe_divide(
        df['variation_resultat_net_N-1'], 
        df['HN_RésultatNet'].abs() + 1e-8  # Évite division par zéro
    )
    df['taux_croissance_RN_N-2'] = safe_divide(
        df['variation_resultat_net_N-2'], 
        df['HN_RésultatNet'].abs() + 1e-8
    )

    df['ratio_liquidite'] = safe_divide(df['DA_TresorerieActive'], df['DL_DettesCourtTerme'])
    df['ratio_stabilite_inv'] = safe_divide(1, df['anciennete'].clip(lower=1))
    df['proxy_actif_taux'] = safe_divide(
        df['DA_TresorerieActive'] - df['DL_DettesCourtTerme'], 
        df['anciennete'].clip(lower=1)
    )
    df['HN_RésultatNet_log'] = safe_log1p(df['HN_RésultatNet'])
    df['FR_ResultatExceptionnel_log'] = safe_log1p(df['FR_ResultatExceptionnel'])
    df['flag_exceptionnel'] = (df['FR_ResultatExceptionnel'] != 0).astype(int)
    df['secteur_NAF_2chiffres'] = df['activitePrincipaleUniteLegale'].astype(str).str[:2]
    df['DL_DettesCourtTerme_log'] = safe_log1p(df['DL_DettesCourtTerme'])
    df['CJCK_TotalActifBrut_log'] = safe_log1p(df['CJCK_TotalActifBrut'])
    df['ratio_dette_ct_vs_actif'] = safe_divide(df['DL_DettesCourtTerme'], df['CJCK_TotalActifBrut'])
    df['ratio_tresorerie_vs_dette_ct'] = safe_divide(df['DA_TresorerieActive'], df['DL_DettesCourtTerme'])
    
    # Targets
    df['target_is_profit'] = (df[TARGET_RN_NPLUS1] > 0).astype(int)
    df['target_magnitude_log'] = np.log1p(np.abs(df[TARGET_RN_NPLUS1]))

    # Sélection des colonnes
    cols_to_keep = FINAL_FEATURE_WHITELIST + [
        TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice'
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Nettoyage
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log'])
    
    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    print(f"✅ Lignes après nettoyage: {len(df)}")
    return df

# ==================================================
## 2. SPLIT TEMPOREL (CORRIGÉ)
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    """Split temporel avec sauvegarde des index."""
    
    df_sorted = df.sort_values(by='AnneeClotureExercice').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    
    # Colonnes à retirer de X
    cols_to_remove = [TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice']
    X = df_sorted.drop(columns=cols_to_remove, errors='ignore')
    
    if X.shape[1] != 19:
        print(f"❌ ERREUR: {X.shape[1]} colonnes au lieu de 19")
        print(f"Colonnes trouvées: {X.columns.tolist()}")
        sys.exit(1)

    Y_reg_log = df_sorted['target_magnitude_log']
    Y_cls = df_sorted['target_is_profit']
    Y_raw = df_sorted[TARGET_RN_NPLUS1]  # ✅ Sauvegarde de Y brut
    
    # Split par position
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    Y_reg_log_train = Y_reg_log.iloc[:split_point].copy()
    Y_cls_train = Y_cls.iloc[:split_point].copy()
    Y_cls_test = Y_cls.iloc[split_point:].copy()
    Y_raw_test = Y_raw.iloc[split_point:].copy()  # ✅ Y brut pour le test

    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_test

# ==================================================
## 3. CLASSIFICATION (CORRIGÉ)
# ==================================================

def train_and_evaluate_cls(X_train, X_test, Y_train, Y_test):
    """Classification avec retour de l'encodeur."""
    
    cls_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
    X_train_encoded = cls_encoder.fit_transform(X_train, Y_train)
    X_test_encoded = cls_encoder.transform(X_test)

    print("\n🔵 [CLASSIFICATION] Entraînement du modèle de signe...")
    cls_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc',
        n_estimators=700, 
        learning_rate=0.03, 
        max_depth=7,
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        subsample=0.8, 
        colsample_bytree=0.8
    )
    cls_model.fit(X_train_encoded, Y_train)

    cls_pred = cls_model.predict(X_test_encoded)
    cls_accuracy = accuracy_score(Y_test, cls_pred)

    print(f"✅ Accuracy Classification: {cls_accuracy:.4f}")
    
    return cls_model, cls_encoder  # ✅ Retour de l'encodeur

# ==================================================
## 4. RÉGRESSION (CORRIGÉ)
# ==================================================

def train_and_evaluate_reg(X_train, Y_train):
    """Cross-validation avec Target Encoding propre."""
    
    print("\n🟢 [RÉGRESSION] Cross-Validation (CV=5)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores_cv = []

    base_regressor = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1500, 
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7, 
        colsample_bytree=0.7, 
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )

    # ✅ CV avec encodage indépendant par fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        Y_tr = Y_train.iloc[train_idx]
        Y_val = Y_train.iloc[val_idx]
        
        # Encodeur fitté uniquement sur le fold de train
        fold_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
        X_tr_enc = fold_encoder.fit_transform(X_tr, Y_tr)
        X_val_enc = fold_encoder.transform(X_val)
        
        fold_model = xgb.XGBRegressor(**base_regressor.get_params())
        fold_model.fit(X_tr_enc, Y_tr)
        
        pred = fold_model.predict(X_val_enc)
        r2 = r2_score(Y_val, pred)
        r2_scores_cv.append(r2)
        print(f"  Fold {fold}: R² = {r2:.4f}")

    print(f"\n📈 R² moyen (CV): {np.mean(r2_scores_cv):.4f} ± {np.std(r2_scores_cv):.4f}")

    # ✅ Entraînement final sur tout le train
    final_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
    X_train_enc = final_encoder.fit_transform(X_train, Y_train)
    
    final_model = xgb.XGBRegressor(**base_regressor.get_params())
    final_model.fit(X_train_enc, Y_train)
    
    return final_model, final_encoder

# ==================================================
## 5. ÉVALUATION GLOBALE (CORRIGÉ)
# ==================================================

def evaluate_global(cls_model, cls_encoder, reg_model, reg_encoder, 
                    X_test, Y_cls_test, Y_raw_test):
    """Évaluation finale SANS data leakage."""
    
    print("\n🎯 [ÉVALUATION] Prédictions sur le test set...")
    
    # ✅ Utilisation des encodeurs DÉJÀ entraînés
    X_test_cls_enc = cls_encoder.transform(X_test)
    X_test_reg_enc = reg_encoder.transform(X_test)
    
    # Prédiction du signe
    pred_sign = cls_model.predict(X_test_cls_enc)
    pred_sign = np.where(pred_sign == 1, 1, -1)
    
    # Prédiction de la magnitude
    pred_magnitude_log = reg_model.predict(X_test_reg_enc)
    pred_magnitude = np.expm1(pred_magnitude_log)
    pred_magnitude = np.maximum(pred_magnitude, 0)  # Pas de magnitude négative
    
    # Combinaison finale
    final_pred = pred_sign * pred_magnitude

    # Métriques
    mae = mean_absolute_error(Y_raw_test, final_pred)
    r2 = r2_score(Y_raw_test, final_pred)

    print("\n" + "="*50)
    print("📊 PERFORMANCE FINALE (Test Set)")
    print("="*50)
    print(f"MAE : {mae:,.2f} €")
    print(f"R²  : {r2:.4f}")
    print("="*50)
    
    # Plots
    plot_feature_importance(reg_model, "feature_importance.png")
    plot_residuals(Y_raw_test, final_pred, "residuals.png")
    
    return final_pred

# ==================================================
## 6. PLOTS
# ==================================================

def plot_feature_importance(model, filename):
    """Plot de l'importance des features."""
    try:
        importance = model.get_booster().get_score(importance_type='gain')
        df_imp = pd.DataFrame(
            list(importance.items()), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 20 Features - Importance (Gain)')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Plot sauvegardé: {filename}")
    except Exception as e:
        print(f"⚠️ Erreur plot importance: {e}")

def plot_residuals(y_true, y_pred, filename):
    """Plot des résidus."""
    try:
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.3, s=10, c='steelblue')
        
        # Ligne de référence parfaite
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
        
        plt.xlabel('Valeur Réelle (€)')
        plt.ylabel('Valeur Prédite (€)')
        plt.title('Analyse des Résidus: Prédiction vs Réalité')
        plt.xscale('symlog')
        plt.yscale('symlog')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Plot sauvegardé: {filename}")
    except Exception as e:
        print(f"⚠️ Erreur plot résidus: {e}")

# ==================================================
## 7. PIPELINE PRINCIPAL
# ==================================================

def main():
    print("="*60)
    print("🚀 PIPELINE ML - PRÉDICTION RÉSULTAT NET (Y_RN)")
    print("="*60)
    
    try:
        # 1. Chargement et feature engineering
        df_raw = load_data(DATA_PATH)
        df_processed = feature_engineering(df_raw)
        
        # 2. Split temporel
        X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_test = split_data(df_processed)
        
        # 3. Classification
        cls_model, cls_encoder = train_and_evaluate_cls(X_train, X_test, Y_cls_train, Y_cls_test)
        
        # 4. Régression
        reg_model, reg_encoder = train_and_evaluate_reg(X_train, Y_reg_log_train)
        
        # 5. Évaluation finale
        final_predictions = evaluate_global(
            cls_model, cls_encoder, reg_model, reg_encoder,
            X_test, Y_cls_test, Y_raw_test
        )
        
        print("\n✅ Pipeline terminé avec succès!")

    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()