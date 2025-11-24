import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
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
## 0. CONFIGURATION - BASELINE QUI FONCTIONNE + AJUSTEMENTS MODÉRÉS
# ==================================================

DATA_PATH = 'Data/processed/sirene_final2.parquet'
TARGET_RN_NPLUS1 = 'Y_RN' 
RANDOM_SEED = 42

TARGET_ENCODING_COLS = ['departement', 'secteur_NAF_2chiffres']

# Features : baseline + seulement quelques ajouts ciblés
FINAL_FEATURE_WHITELIST = [
    # Features baseline (19)
    'ratio_rentabilite_nette', 'ratio_endettement', 'ratio_tresorerie', 
    'ratio_resultat_financier', 'ratio_resultat_exceptionnel', 
    'ratio_liquidite', 'ratio_stabilite_inv', 'proxy_actif_taux',
    'HN_RésultatNet_log', 'FR_ResultatExceptionnel_log', 
    'DL_DettesCourtTerme_log', 'CJCK_TotalActifBrut_log',
    'ratio_dette_ct_vs_actif', 'ratio_tresorerie_vs_dette_ct', 
    'flag_exceptionnel', 
    'taux_croissance_RN_N-1', 'taux_croissance_RN_N-2', 
    'departement', 'secteur_NAF_2chiffres',
    # 🆕 Seulement 3 nouvelles features SIMPLES et efficaces
    'interaction_liquidite_endettement',
    'RN_volatility_log',
    'flag_croissance',
]

COLS_TO_WINSORIZE = [
    TARGET_RN_NPLUS1, 'HN_RésultatNet', 'CJCK_TotalActifBrut', 
    'DL_DettesCourtTerme', 'FR_ResultatExceptionnel'
]
# Garder la winsorisation originale qui fonctionnait
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
## 1. FEATURE ENGINEERING - BASELINE + AJOUTS CIBLÉS
# ==================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering prudent : baseline + 3 features testées."""
    
    print("\n🔧 Application de la Winsorisation (2.5%-97.5%)...")
    for col in COLS_TO_WINSORIZE:
        if col in df.columns:
            lower_bound = df[col].quantile(WINSOR_LOW)
            upper_bound = df[col].quantile(WINSOR_HIGH)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # --- FEATURES BASELINE (qui fonctionnaient) ---
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
    
    # 🆕 SEULEMENT 3 nouvelles features SIMPLES
    print("🆕 Ajout de 3 nouvelles features ciblées...")
    
    # 1. Interaction liquidité-endettement
    df['interaction_liquidite_endettement'] = df['ratio_liquidite'] * df['ratio_endettement']
    
    # 2. Volatilité des résultats (log)
    df['RN_volatility'] = np.abs(
        df['variation_resultat_net_N-1'] - df['variation_resultat_net_N-2']
    )
    df['RN_volatility_log'] = safe_log1p(df['RN_volatility'])
    
    # 3. Flag croissance continue
    df['flag_croissance'] = (
        (df['taux_croissance_RN_N-1'] > 0) & 
        (df['taux_croissance_RN_N-2'] > 0)
    ).astype(int)
    
    # --- TARGETS ---
    df['target_is_profit'] = (df[TARGET_RN_NPLUS1] > 0).astype(int)
    df['target_magnitude_log'] = np.log1p(np.abs(df[TARGET_RN_NPLUS1]))

    # Sélection
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
    
    print(f"✅ Features: {len([c for c in df.columns if c in FINAL_FEATURE_WHITELIST])} | Lignes: {len(df):,}")
    return df

# ==================================================
## 2. SPLIT TEMPOREL
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    df_sorted = df.sort_values(by='AnneeClotureExercice').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    
    cols_to_remove = [TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice']
    X = df_sorted.drop(columns=cols_to_remove, errors='ignore')
    
    expected_features = len(FINAL_FEATURE_WHITELIST)
    if X.shape[1] != expected_features:
        print(f"⚠️  {X.shape[1]} colonnes au lieu de {expected_features}")

    Y_reg_log = df_sorted['target_magnitude_log']
    Y_cls = df_sorted['target_is_profit']
    Y_raw = df_sorted[TARGET_RN_NPLUS1]
    
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    Y_reg_log_train = Y_reg_log.iloc[:split_point].copy()
    Y_cls_train = Y_cls.iloc[:split_point].copy()
    Y_cls_test = Y_cls.iloc[split_point:].copy()
    Y_raw_test = Y_raw.iloc[split_point:].copy()

    print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_test

# ==================================================
## 3. CLASSIFICATION - BASELINE AMÉLIORÉ
# ==================================================

def train_and_evaluate_cls(X_train, X_test, Y_train, Y_test):
    """Classification avec smoothing légèrement augmenté."""
    
    # Smoothing modéré (10→15 au lieu de 25)
    cls_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=15)
    X_train_encoded = cls_encoder.fit_transform(X_train, Y_train)
    X_test_encoded = cls_encoder.transform(X_test)

    print("\n🔵 [CLASSIFICATION] Entraînement...")
    cls_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc',
        n_estimators=700, 
        learning_rate=0.03,  # Baseline
        max_depth=7,  # Baseline
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        subsample=0.8,  # Baseline
        colsample_bytree=0.8,  # Baseline
    )
    cls_model.fit(X_train_encoded, Y_train)

    cls_pred = cls_model.predict(X_test_encoded)
    cls_accuracy = accuracy_score(Y_test, cls_pred)
    print(f"✅ Accuracy: {cls_accuracy:.4f}")
    
    return cls_model, cls_encoder

# ==================================================
## 4. RÉGRESSION - BASELINE AVEC AJUSTEMENTS GRADUELS
# ==================================================

def train_and_evaluate_reg(X_train, Y_train):
    """Régression avec régularisation MODÉRÉE et early stopping."""
    
    print("\n🟢 [RÉGRESSION] Cross-Validation (CV=5)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_scores_cv = []

    # Hyperparamètres : baseline + ajustements MODÉRÉS
    base_regressor = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=2000,  # Plus d'arbres pour early stopping
        max_depth=5,  # 🔧 Légèrement réduit (6→5)
        learning_rate=0.025,  # 🔧 Légèrement réduit (0.03→0.025)
        subsample=0.7,  # Baseline
        colsample_bytree=0.7,  # Baseline
        reg_alpha=0.7,  # 🔧 Légèrement augmenté (0.5→0.7)
        reg_lambda=0.7,  # 🔧 Légèrement augmenté (0.5→0.7)
        min_child_weight=3,  # 🆕 Mais modéré
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        early_stopping_rounds=100,  # 🆕 Early stopping
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        Y_tr = Y_train.iloc[train_idx]
        Y_val = Y_train.iloc[val_idx]
        
        # Smoothing modéré
        fold_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=15)
        X_tr_enc = fold_encoder.fit_transform(X_tr, Y_tr)
        X_val_enc = fold_encoder.transform(X_val)
        
        fold_model = xgb.XGBRegressor(**base_regressor.get_params())
        
        # Fit avec early stopping
        fold_model.fit(
            X_tr_enc, Y_tr,
            eval_set=[(X_val_enc, Y_val)],
            verbose=False
        )
        
        pred = fold_model.predict(X_val_enc)
        r2 = r2_score(Y_val, pred)
        r2_scores_cv.append(r2)
        print(f"  Fold {fold}: R² = {r2:.4f} (best_iteration={fold_model.best_iteration})")

    print(f"\n📈 R² moyen (CV): {np.mean(r2_scores_cv):.4f} ± {np.std(r2_scores_cv):.4f}")

    # Entraînement final
    final_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=15)
    X_train_enc = final_encoder.fit_transform(X_train, Y_train)
    
    final_model = xgb.XGBRegressor(**base_regressor.get_params())
    final_model.set_params(early_stopping_rounds=None)  # Pas d'early stopping sur final
    final_model.fit(X_train_enc, Y_train, verbose=False)
    
    return final_model, final_encoder

# ==================================================
## 5. MÉTRIQUES
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

# ==================================================
## 6. ÉVALUATION GLOBALE
# ==================================================

def evaluate_global(cls_model, cls_encoder, reg_model, reg_encoder, 
                    X_test, Y_cls_test, Y_raw_test):
    """Évaluation finale."""
    
    print("\n🎯 [ÉVALUATION] Prédictions sur le test set...")
    
    X_test_cls_enc = cls_encoder.transform(X_test)
    X_test_reg_enc = reg_encoder.transform(X_test)
    
    # Prédiction du signe
    pred_sign = cls_model.predict(X_test_cls_enc)
    pred_sign = np.where(pred_sign == 1, 1, -1)
    
    # Prédiction de la magnitude
    pred_magnitude_log = reg_model.predict(X_test_reg_enc)
    pred_magnitude = np.expm1(pred_magnitude_log)
    pred_magnitude = np.maximum(pred_magnitude, 0)
    
    # Combinaison
    final_pred = pred_sign * pred_magnitude

    # Métriques
    metrics = calculate_metrics(Y_raw_test, final_pred)
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE FINALE (Test Set)")
    print("="*60)
    print(f"MAE       : {metrics['MAE']:>15,.2f} €")
    print(f"MedAE     : {metrics['MedAE']:>15,.2f} € (médiane)")
    print(f"RMSE      : {metrics['RMSE']:>15,.2f} €")
    print(f"MAPE      : {metrics['MAPE']:>15.2f} %")
    print(f"R²        : {metrics['R²']:>15.4f}")
    print("="*60)
    
    # Analyse des erreurs
    errors = Y_raw_test - final_pred
    analyze_errors(Y_raw_test, final_pred, errors)
    
    # Plots
    create_comprehensive_plots(Y_raw_test, final_pred, errors, reg_model)
    
    return final_pred, metrics

# ==================================================
## 7. ANALYSE DES ERREURS
# ==================================================

def analyze_errors(y_true, y_pred, errors):
    print("\n" + "="*60)
    print("🔍 ANALYSE DÉTAILLÉE DES ERREURS")
    print("="*60)
    
    print("\n📊 Distribution des erreurs (€):")
    print(f"  Minimum     : {errors.min():>15,.0f}")
    print(f"  Q1 (25%)    : {np.percentile(errors, 25):>15,.0f}")
    print(f"  Médiane     : {np.median(errors):>15,.0f}")
    print(f"  Q3 (75%)    : {np.percentile(errors, 75):>15,.0f}")
    print(f"  Maximum     : {errors.max():>15,.0f}")
    print(f"  Écart-type  : {errors.std():>15,.0f}")
    
    abs_errors = np.abs(errors)
    print("\n📊 Erreurs absolues (€):")
    print(f"  Médiane     : {np.median(abs_errors):>15,.0f}")
    print(f"  Moyenne     : {abs_errors.mean():>15,.0f}")
    print(f"  Q90         : {np.percentile(abs_errors, 90):>15,.0f}")
    print(f"  Q95         : {np.percentile(abs_errors, 95):>15,.0f}")
    
    print("\n📊 Analyse par seuils:")
    for threshold in [100_000, 500_000, 1_000_000, 5_000_000]:
        pct = (abs_errors > threshold).mean() * 100
        print(f"  Erreurs > {threshold/1e6:.1f}M€ : {pct:>6.2f}%")
    
    print("\n📊 Performance par signe:")
    mask_profit = y_true > 0
    mask_loss = y_true <= 0
    
    for label, mask in [('Profits', mask_profit), ('Pertes', mask_loss)]:
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            r2 = r2_score(y_true[mask], y_pred[mask])
            mape = calculate_mape(y_true[mask], y_pred[mask])
            print(f"  {label:>8}: MAE={mae:>12,.0f}€  R²={r2:>6.3f}  MAPE={mape:>6.1f}%  (n={mask.sum():,})")
    
    print("="*60)

# ==================================================
## 8. VISUALISATIONS
# ==================================================

def create_comprehensive_plots(y_true, y_pred, errors, model):
    print("\n📈 Génération des visualisations...")
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    create_main_dashboard(y_true, y_pred, errors)
    if model:
        plot_feature_importance(model)
    plot_residuals_detailed(y_true, y_pred, errors)
    plot_distribution_comparison(y_true, y_pred)
    plot_quantile_analysis(y_true, y_pred)
    
    print("✅ Toutes les visualisations générées!")

def create_main_dashboard(y_true, y_pred, errors):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dashboard Principal - Approche Progressive', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=20, c='steelblue', edgecolors='none')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
    ax1.set_xlabel('Valeur Réelle (€)', fontsize=11)
    ax1.set_ylabel('Valeur Prédite (€)', fontsize=11)
    ax1.set_title('Prédictions vs Réalité', fontsize=12, fontweight='bold')
    ax1.set_xscale('symlog')
    ax1.set_yscale('symlog')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(errors / 1e6, bins=100, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Erreur (M€)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Distribution des Erreurs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    abs_errors = np.abs(errors)
    ax3.scatter(np.abs(y_true), abs_errors, alpha=0.3, s=20, c='green', edgecolors='none')
    ax3.set_xlabel('|Valeur Réelle| (€)', fontsize=11)
    ax3.set_ylabel('Erreur Absolue (€)', fontsize=11)
    ax3.set_title('Erreur Absolue vs Magnitude', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    mask = np.abs(y_true) > 1000
    relative_errors = np.abs(errors[mask] / y_true[mask]) * 100
    relative_errors_clipped = np.clip(relative_errors, 0, 200)
    ax4.hist(relative_errors_clipped, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(np.median(relative_errors_clipped), color='red', linestyle='--', lw=2,
                label=f'Médiane: {np.median(relative_errors_clipped):.1f}%')
    ax4.set_xlabel('Erreur Relative (%)', fontsize=11)
    ax4.set_ylabel('Fréquence', fontsize=11)
    ax4.set_title('Distribution des Erreurs Relatives', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_dashboard_principal.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Dashboard")

def plot_feature_importance(model):
    try:
        importance = model.get_booster().get_score(importance_type='gain')
        df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        df_imp = df_imp.sort_values('Importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(df_imp))
        plt.barh(range(len(df_imp)), df_imp['Importance'], color=colors)
        plt.yticks(range(len(df_imp)), df_imp['Feature'])
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.title('Top 20 Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('02_feature_importance.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance")
    except:
        print("  ⚠️ Skip feature importance")

def plot_residuals_detailed(y_true, y_pred, errors):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analyse des Résidus', fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    ax1.scatter(y_pred, errors, alpha=0.3, s=20, c='steelblue', edgecolors='none')
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_xlabel('Valeur Prédite (€)', fontsize=11)
    ax1.set_ylabel('Résidu', fontsize=11)
    ax1.set_title('Résidus vs Prédictions', fontsize=12)
    ax1.set_xscale('symlog')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(errors / errors.std(), dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_residuals_analysis.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Résidus")

def plot_distribution_comparison(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparaison des Distributions', fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    ax1.hist(y_true / 1e6, bins=100, alpha=0.6, label='Réel', color='blue', edgecolor='black')
    ax1.hist(y_pred / 1e6, bins=100, alpha=0.6, label='Prédit', color='red', edgecolor='black')
    ax1.set_xlabel('Résultat Net (M€)', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distributions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    bp = ax2.boxplot([y_true / 1e6, y_pred / 1e6], tick_labels=['Réel', 'Prédit'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Résultat Net (M€)', fontsize=11)
    ax2.set_title('Box Plots', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('04_distribution_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Distributions")

def plot_quantile_analysis(y_true, y_pred):
    n_quantiles = 10
    quantiles = pd.qcut(y_true, q=n_quantiles, labels=False, duplicates='drop')
    
    metrics_by_quantile = []
    for q in range(n_quantiles):
        mask = quantiles == q
        if mask.sum() > 10:
            mae_q = mean_absolute_error(y_true[mask], y_pred[mask])
            try:
                r2_q = r2_score(y_true[mask], y_pred[mask])
            except:
                r2_q = np.nan
            metrics_by_quantile.append({'quantile': q + 1, 'mae': mae_q, 'r2': r2_q})
    
    df_q = pd.DataFrame(metrics_by_quantile)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance par Quantile', fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    ax1.bar(df_q['quantile'], df_q['mae'] / 1e6, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Quantile', fontsize=11)
    ax1.set_ylabel('MAE (M€)', fontsize=11)
    ax1.set_title('MAE par Quantile', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[1]
    colors = ['red' if r2 < 0 else 'green' for r2 in df_q['r2']]
    ax2.bar(df_q['quantile'], df_q['r2'], color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='--', lw=1)
    ax2.set_xlabel('Quantile', fontsize=11)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_title('R² par Quantile', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('05_quantile_analysis.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Quantiles")

# ==================================================
## 9. RECOMMANDATIONS
# ==================================================

def print_summary(metrics, baseline_metrics):
    print("\n" + "="*60)
    print("📊 RÉSUMÉ - APPROCHE PROGRESSIVE")
    print("="*60)
    
    print("\n✅ Ajustements appliqués (MODÉRÉS):")
    print("  • 3 nouvelles features ciblées (22 au total)")
    print("  • Smoothing: 10 → 15 (modéré)")
    print("  • Max depth: 6 → 5 (léger)")
    print("  • Learning rate: 0.03 → 0.025 (léger)")
    print("  • Régularisation: L1/L2 0.5 → 0.7 (modérée)")
    print("  • Early stopping: 100 rounds")
    print("  • Min child weight: 3 (nouveau)")
    
    print("\n📊 Comparaison Baseline → Progressive:")
    for metric in ['MAE', 'R²', 'MAPE']:
        old = baseline_metrics.get(metric, 0)
        new = metrics.get(metric, 0)
        if metric == 'MAE':
            delta = ((new - old) / old * 100) if old != 0 else 0
            symbol = "✅" if delta < 0 else "⚠️"
            print(f"  {symbol} {metric:>6}: {old:>12,.0f}€ → {new:>12,.0f}€  ({delta:+.1f}%)")
        elif metric == 'MAPE':
            delta = new - old
            symbol = "✅" if delta < 0 else "⚠️"
            print(f"  {symbol} {metric:>6}: {old:>12,.1f}% → {new:>12,.1f}%  ({delta:+.1f} pts)")
        else:
            delta = new - old
            symbol = "✅" if delta > 0 else "⚠️"
            print(f"  {symbol} {metric:>6}: {old:>12.4f} → {new:>12.4f}  ({delta:+.4f})")
    
    print("\n🎯 Prochaines étapes si résultats positifs:")
    print("  1. Ajouter progressivement plus de features")
    print("  2. Tester LightGBM (souvent meilleur sur tabular)")
    print("  3. Grid search sur hyperparamètres")
    print("  4. Essayer stacking léger (XGB + Ridge)")
    print("="*60)

# ==================================================
## 10. PIPELINE PRINCIPAL
# ==================================================

def main():
    print("="*60)
    print("🚀 PIPELINE ML - APPROCHE PROGRESSIVE")
    print("="*60)
    
    try:
        df_raw = load_data(DATA_PATH)
        df_processed = feature_engineering(df_raw)
        
        X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_test = split_data(df_processed)
        
        cls_model, cls_encoder = train_and_evaluate_cls(X_train, X_test, Y_cls_train, Y_cls_test)
        reg_model, reg_encoder = train_and_evaluate_reg(X_train, Y_reg_log_train)
        
        final_predictions, metrics = evaluate_global(
            cls_model, cls_encoder, reg_model, reg_encoder,
            X_test, Y_cls_test, Y_raw_test
        )
        
        # Baseline pour comparaison
        baseline = {'MAE': 352099.47, 'R²': 0.5644, 'MAPE': 181.58}
        print_summary(metrics, baseline)
        
        print("\n✅ Pipeline terminé!")
        print("📁 5 visualisations générées (01-05_*.png)")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()