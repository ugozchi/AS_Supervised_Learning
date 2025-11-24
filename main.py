import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import category_encoders as ce  
import pandas as pd  
import warnings
import os
import polars as pl
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

warnings.filterwarnings('ignore')

# ==================================================
## 0. CONFIGURATION - RÉGRESSION DIRECTE AVEC LIGHTGBM
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
    # Nouvelles features
    'interaction_liquidite_endettement',
    'RN_volatility_log',
    'flag_croissance',
]

COLS_TO_WINSORIZE = [
    TARGET_RN_NPLUS1, 'HN_RésultatNet', 'CJCK_TotalActifBrut', 
    'DL_DettesCourtTerme', 'FR_ResultatExceptionnel'
]
WINSOR_LOW = 0.025
WINSOR_HIGH = 0.975


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERREUR: Fichier non trouvé: {file_path}")
    df_pl = pl.read_parquet(file_path)
    return df_pl.to_pandas()

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
## 1. FEATURE ENGINEERING
# ==================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔧 Winsorisation (2.5%-97.5%)...")
    for col in COLS_TO_WINSORIZE:
        if col in df.columns:
            lower_bound = df[col].quantile(WINSOR_LOW)
            upper_bound = df[col].quantile(WINSOR_HIGH)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    df['taux_croissance_RN_N-1'] = safe_divide(
        df['variation_resultat_net_N-1'], 
        df['HN_RésultatNet'].abs() + 1e-8
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
    
    print("🆕 Ajout de 3 features...")
    df['interaction_liquidite_endettement'] = df['ratio_liquidite'] * df['ratio_endettement']
    df['RN_volatility'] = np.abs(
        df['variation_resultat_net_N-1'] - df['variation_resultat_net_N-2']
    )
    df['RN_volatility_log'] = safe_log1p(df['RN_volatility'])
    df['flag_croissance'] = (
        (df['taux_croissance_RN_N-1'] > 0) & 
        (df['taux_croissance_RN_N-2'] > 0)
    ).astype(int)
    
    # 🆕 TARGET : Transformation SYMLOG pour régression directe
    # Au lieu de séparer signe/magnitude, on prédit directement avec symlog
    df['target_symlog'] = safe_log1p(df[TARGET_RN_NPLUS1])
    
    cols_to_keep = FINAL_FEATURE_WHITELIST + [
        TARGET_RN_NPLUS1, 'target_symlog', 'AnneeClotureExercice'
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_RN_NPLUS1, 'target_symlog'])
    
    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')  # LightGBM gère les catégories nativement
    
    print(f"✅ Features: {len([c for c in df.columns if c in FINAL_FEATURE_WHITELIST])} | Lignes: {len(df):,}")
    return df

# ==================================================
## 2. SPLIT TEMPOREL
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    df_sorted = df.sort_values(by='AnneeClotureExercice').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    
    cols_to_remove = [TARGET_RN_NPLUS1, 'target_symlog', 'AnneeClotureExercice']
    X = df_sorted.drop(columns=cols_to_remove, errors='ignore')
    
    Y_symlog = df_sorted['target_symlog']
    Y_raw = df_sorted[TARGET_RN_NPLUS1]
    
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    Y_symlog_train = Y_symlog.iloc[:split_point].copy()
    Y_symlog_test = Y_symlog.iloc[split_point:].copy()
    Y_raw_train = Y_raw.iloc[:split_point].copy()
    Y_raw_test = Y_raw.iloc[split_point:].copy()

    print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, Y_symlog_train, Y_symlog_test, Y_raw_train, Y_raw_test

# ==================================================
## 3. RÉGRESSION DIRECTE AVEC LIGHTGBM + SAMPLE WEIGHTS
# ==================================================

def calculate_sample_weights(y_true, mode='balanced'):
    """
    Calcule les poids pour équilibrer l'importance des petites et grandes valeurs.
    Mode 'balanced' : poids inversement proportionnel à la magnitude.
    """
    if mode == 'balanced':
        # Plus de poids aux petites valeurs (pour améliorer MAPE)
        weights = 1.0 / (np.abs(y_true) + 10000)  # +10k€ pour éviter division par 0
        # Normaliser entre 0.5 et 2.0
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.5, 2.0)
        return weights
    else:
        return np.ones(len(y_true))

def train_lightgbm_with_cv(X_train, Y_symlog_train, Y_raw_train):
    """
    Entraîne LightGBM avec CV et sample weights optimisés pour MAPE.
    """
    
    print("\n🟢 [LIGHTGBM] Régression directe avec CV (5 folds)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # 🔥 Paramètres LightGBM optimisés
    params = {
        'objective': 'regression',
        'metric': 'mae',  # Optimiser pour MAE (corrélé avec MAPE)
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Complexité modérée
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_child_samples': 20,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    r2_scores = []
    mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        Y_tr = Y_symlog_train.iloc[train_idx]
        Y_val = Y_symlog_train.iloc[val_idx]
        Y_raw_tr = Y_raw_train.iloc[train_idx]
        Y_raw_val = Y_raw_train.iloc[val_idx]
        
        # Target encoding
        fold_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=15)
        X_tr_enc = fold_encoder.fit_transform(X_tr, Y_tr)
        X_val_enc = fold_encoder.transform(X_val)
        
        # 🔥 Sample weights pour améliorer MAPE
        sample_weights = calculate_sample_weights(Y_raw_tr, mode='balanced')
        
        # Entraînement
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr_enc, Y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_val_enc, Y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        # Prédictions en symlog
        pred_symlog = model.predict(X_val_enc)
        
        # Conversion en valeurs brutes
        pred_raw = np.sign(pred_symlog) * (np.expm1(np.abs(pred_symlog)))
        
        # Métriques
        r2 = r2_score(Y_raw_val, pred_raw)
        mae = mean_absolute_error(Y_raw_val, pred_raw)
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        
        print(f"  Fold {fold}: R²={r2:.4f} | MAE={mae:,.0f}€ | best_iter={model.best_iteration_}")
    
    print(f"\n📈 Moyennes CV:")
    print(f"  R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"  MAE : {np.mean(mae_scores):,.0f}€ ± {np.std(mae_scores):,.0f}€")
    
    # 🔨 Entraînement final sur tout le train
    print("\n🔨 Entraînement du modèle final...")
    final_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=15)
    X_train_enc = final_encoder.fit_transform(X_train, Y_symlog_train)
    
    sample_weights_final = calculate_sample_weights(Y_raw_train, mode='balanced')
    
    final_model = lgb.LGBMRegressor(**params)
    final_model.set_params(n_estimators=int(np.mean([model.best_iteration_ for model in [final_model]])) if hasattr(final_model, 'best_iteration_') else 1000)
    final_model.fit(
        X_train_enc, Y_symlog_train,
        sample_weight=sample_weights_final
    )
    
    return final_model, final_encoder

# ==================================================
## 4. ÉVALUATION
# ==================================================

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    mask = np.abs(y_true) > 1000
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    medae = np.median(np.abs(y_true - y_pred))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'MedAE': medae
    }

def evaluate_model(model, encoder, X_test, Y_raw_test):
    print("\n🎯 [ÉVALUATION] Test set...")
    
    # Encodage
    X_test_enc = encoder.transform(X_test)
    
    # Prédiction en symlog
    pred_symlog = model.predict(X_test_enc)
    
    # Conversion en valeurs brutes
    final_pred = np.sign(pred_symlog) * (np.expm1(np.abs(pred_symlog)))
    
    # Métriques
    metrics = calculate_metrics(Y_raw_test, final_pred)
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE FINALE (Test Set - LightGBM Direct)")
    print("="*60)
    print(f"MAE       : {metrics['MAE']:>15,.2f} €")
    print(f"MedAE     : {metrics['MedAE']:>15,.2f} € (médiane)")
    print(f"RMSE      : {metrics['RMSE']:>15,.2f} €")
    print(f"MAPE      : {metrics['MAPE']:>15.2f} %")
    print(f"R²        : {metrics['R²']:>15.4f}")
    print("="*60)
    
    # Analyse détaillée
    errors = Y_raw_test - final_pred
    analyze_errors(Y_raw_test, final_pred, errors)
    
    # Visualisations
    create_comprehensive_plots(Y_raw_test, final_pred, errors, model)
    
    return final_pred, metrics

# ==================================================
## 5. ANALYSE DES ERREURS
# ==================================================

def analyze_errors(y_true, y_pred, errors):
    print("\n" + "="*60)
    print("🔍 ANALYSE DÉTAILLÉE DES ERREURS")
    print("="*60)
    
    print("\n📊 Distribution des erreurs (€):")
    print(f"  Min      : {errors.min():>15,.0f}")
    print(f"  Q25      : {np.percentile(errors, 25):>15,.0f}")
    print(f"  Médiane  : {np.median(errors):>15,.0f}")
    print(f"  Q75      : {np.percentile(errors, 75):>15,.0f}")
    print(f"  Max      : {errors.max():>15,.0f}")
    print(f"  Std      : {errors.std():>15,.0f}")
    
    abs_errors = np.abs(errors)
    print("\n📊 Erreurs absolues:")
    print(f"  Médiane  : {np.median(abs_errors):>15,.0f} €")
    print(f"  Moyenne  : {abs_errors.mean():>15,.0f} €")
    print(f"  Q90      : {np.percentile(abs_errors, 90):>15,.0f} €")
    print(f"  Q95      : {np.percentile(abs_errors, 95):>15,.0f} €")
    
    print("\n📊 Par seuils:")
    for t in [100_000, 500_000, 1_000_000, 5_000_000]:
        pct = (abs_errors > t).mean() * 100
        print(f"  > {t/1e6:.1f}M€ : {pct:>5.2f}%")
    
    print("\n📊 Par signe:")
    for label, mask in [('Profits', y_true > 0), ('Pertes', y_true <= 0)]:
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            r2 = r2_score(y_true[mask], y_pred[mask])
            mape = calculate_mape(y_true[mask], y_pred[mask])
            print(f"  {label:>8}: MAE={mae:>10,.0f}€ | R²={r2:>6.3f} | MAPE={mape:>6.1f}% | n={mask.sum():,}")
    
    print("="*60)

# ==================================================
## 6. VISUALISATIONS
# ==================================================

def create_comprehensive_plots(y_true, y_pred, errors, model):
    print("\n📈 Génération des visualisations...")
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    create_dashboard(y_true, y_pred, errors)
    plot_feature_importance_lgb(model)
    plot_residuals(y_true, y_pred, errors)
    
    print("✅ Visualisations générées!")

def create_dashboard(y_true, y_pred, errors):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LightGBM - Régression Directe', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=15, c='steelblue')
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lim, lim, 'r--', lw=2)
    ax1.set_xlabel('Réel (€)')
    ax1.set_ylabel('Prédit (€)')
    ax1.set_title('Prédictions vs Réalité')
    ax1.set_xscale('symlog')
    ax1.set_yscale('symlog')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(errors / 1e6, bins=100, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Erreur (M€)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Erreurs')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    abs_errors = np.abs(errors)
    ax3.scatter(np.abs(y_true), abs_errors, alpha=0.3, s=15, c='green')
    ax3.set_xlabel('|Réel| (€)')
    ax3.set_ylabel('Erreur Absolue (€)')
    ax3.set_title('Erreur vs Magnitude')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    mask = np.abs(y_true) > 1000
    rel_errors = np.clip(np.abs(errors[mask] / y_true[mask]) * 100, 0, 200)
    ax4.hist(rel_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(np.median(rel_errors), color='red', linestyle='--', lw=2,
                label=f'Médiane: {np.median(rel_errors):.1f}%')
    ax4.set_xlabel('Erreur Relative (%)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('Erreurs Relatives')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_dashboard_lightgbm.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Dashboard")

def plot_feature_importance_lgb(model):
    try:
        importance = model.feature_importances_
        feature_names = model.feature_name_
        
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(df_imp))
        plt.barh(range(len(df_imp)), df_imp['Importance'], color=colors)
        plt.yticks(range(len(df_imp)), df_imp['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Features - LightGBM', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('02_feature_importance_lgb.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance")
    except:
        print("  ⚠️ Skip feature importance")

def plot_residuals(y_true, y_pred, errors):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(y_pred, errors, alpha=0.3, s=15, c='steelblue')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Prédiction (€)')
    ax.set_ylabel('Résidu (€)')
    ax.set_title('Analyse des Résidus', fontsize=14, fontweight='bold')
    ax.set_xscale('symlog')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_residuals_lgb.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Résidus")

# ==================================================
## 7. RÉSUMÉ
# ==================================================

def print_summary(metrics, baseline):
    print("\n" + "="*60)
    print("📊 RÉSUMÉ - LIGHTGBM RÉGRESSION DIRECTE")
    print("="*60)
    
    print("\n🆕 Changements majeurs:")
    print("  • Abandon du modèle hybride Classification/Régression")
    print("  • Régression DIRECTE avec transformation symlog")
    print("  • LightGBM au lieu de XGBoost")
    print("  • Sample weights pour améliorer MAPE")
    print("  • Optimisation objective='mae' au lieu de 'mse'")
    
    print("\n📊 Baseline (Hybride XGB) → LightGBM Direct:")
    for metric in ['MAE', 'R²', 'MAPE']:
        old = baseline.get(metric, 0)
        new = metrics.get(metric, 0)
        if metric == 'MAE':
            delta = ((new - old) / old * 100) if old != 0 else 0
            symbol = "✅" if delta < -5 else "⚠️" if delta < 0 else "❌"
            print(f"  {symbol} {metric:>6}: {old:>10,.0f}€ → {new:>10,.0f}€  ({delta:+.1f}%)")
        elif metric == 'MAPE':
            delta = new - old
            symbol = "✅" if delta < -10 else "⚠️" if delta < 0 else "❌"
            print(f"  {symbol} {metric:>6}: {old:>10.1f}% → {new:>10.1f}%  ({delta:+.1f} pts)")
        else:
            delta = new - old
            symbol = "✅" if delta > 0.02 else "⚠️" if delta > 0 else "❌"
            print(f"  {symbol} {metric:>6}: {old:>10.4f} → {new:>10.4f}  ({delta:+.4f})")
    
    print("\n🎯 Avantages de cette approche:")
    print("  • Meilleure gestion des PERTES (pas de séparation artificielle)")
    print("  • LightGBM plus rapide et souvent plus précis")
    print("  • Sample weights améliore MAPE sur petites valeurs")
    print("  • Pas de propagation d'erreur (classification → régression)")
    
    print("="*60)

# ==================================================
## 8. PIPELINE PRINCIPAL
# ==================================================

def main():
    print("="*60)
    print("🚀 LIGHTGBM - RÉGRESSION DIRECTE")
    print("="*60)
    
    try:
        df_raw = load_data(DATA_PATH)
        df_processed = feature_engineering(df_raw)
        
        X_train, X_test, Y_symlog_train, Y_symlog_test, Y_raw_train, Y_raw_test = split_data(df_processed)
        
        model, encoder = train_lightgbm_with_cv(X_train, Y_symlog_train, Y_raw_train)
        
        final_pred, metrics = evaluate_model(model, encoder, X_test, Y_raw_test)
        
        baseline = {'MAE': 352099, 'R²': 0.5644, 'MAPE': 181.58}
        print_summary(metrics, baseline)
        
        print("\n✅ Pipeline LightGBM terminé!")
        print("📁 3 visualisations: 01-03_*.png")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()