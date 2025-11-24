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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================================================
## 0. CONFIGURATION
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
    print("\n🔧 Application de la Winsorisation...")
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
    
    df['target_is_profit'] = (df[TARGET_RN_NPLUS1] > 0).astype(int)
    df['target_magnitude_log'] = np.log1p(np.abs(df[TARGET_RN_NPLUS1]))

    cols_to_keep = FINAL_FEATURE_WHITELIST + [
        TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice'
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log'])
    
    for col in TARGET_ENCODING_COLS:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    print(f"✅ Lignes après nettoyage: {len(df)}")
    return df

# ==================================================
## 2. SPLIT TEMPOREL
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    df_sorted = df.sort_values(by='AnneeClotureExercice').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    
    cols_to_remove = [TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice']
    X = df_sorted.drop(columns=cols_to_remove, errors='ignore')
    
    if X.shape[1] != 19:
        print(f"❌ ERREUR: {X.shape[1]} colonnes au lieu de 19")
        sys.exit(1)

    Y_reg_log = df_sorted['target_magnitude_log']
    Y_cls = df_sorted['target_is_profit']
    Y_raw = df_sorted[TARGET_RN_NPLUS1]
    
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    Y_reg_log_train = Y_reg_log.iloc[:split_point].copy()
    Y_cls_train = Y_cls.iloc[:split_point].copy()
    Y_cls_test = Y_cls.iloc[split_point:].copy()
    Y_raw_test = Y_raw.iloc[split_point:].copy()

    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_test

# ==================================================
## 3. CLASSIFICATION
# ==================================================

def train_and_evaluate_cls(X_train, X_test, Y_train, Y_test):
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
    return cls_model, cls_encoder

# ==================================================
## 4. RÉGRESSION
# ==================================================

def train_and_evaluate_reg(X_train, Y_train):
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

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        Y_tr = Y_train.iloc[train_idx]
        Y_val = Y_train.iloc[val_idx]
        
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

    final_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
    X_train_enc = final_encoder.fit_transform(X_train, Y_train)
    
    final_model = xgb.XGBRegressor(**base_regressor.get_params())
    final_model.fit(X_train_enc, Y_train)
    
    return final_model, final_encoder

# ==================================================
## 5. MÉTRIQUES AVANCÉES
# ==================================================

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """MAPE robuste qui gère les valeurs proches de zéro."""
    # Exclure les valeurs très proches de zéro
    mask = np.abs(y_true) > 1000  # Seuil de 1000€
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100

def calculate_metrics(y_true, y_pred):
    """Calcule toutes les métriques de performance."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    # Médiane de l'erreur absolue (plus robuste que MAE)
    medae = np.median(np.abs(y_true - y_pred))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'MedAE': medae
    }

# ==================================================
## 6. ÉVALUATION GLOBALE + ANALYSES
# ==================================================

def evaluate_global(cls_model, cls_encoder, reg_model, reg_encoder, 
                    X_test, Y_cls_test, Y_raw_test):
    print("\n🎯 [ÉVALUATION] Prédictions sur le test set...")
    
    X_test_cls_enc = cls_encoder.transform(X_test)
    X_test_reg_enc = reg_encoder.transform(X_test)
    
    pred_sign = cls_model.predict(X_test_cls_enc)
    pred_sign = np.where(pred_sign == 1, 1, -1)
    
    pred_magnitude_log = reg_model.predict(X_test_reg_enc)
    pred_magnitude = np.expm1(pred_magnitude_log)
    pred_magnitude = np.maximum(pred_magnitude, 0)
    
    final_pred = pred_sign * pred_magnitude

    # Calcul des métriques
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
    
    # Génération des plots
    create_comprehensive_plots(Y_raw_test, final_pred, errors, reg_model)
    
    return final_pred, metrics

# ==================================================
## 7. ANALYSE DES ERREURS
# ==================================================

def analyze_errors(y_true, y_pred, errors):
    """Analyse détaillée de la distribution des erreurs."""
    
    print("\n" + "="*60)
    print("🔍 ANALYSE DÉTAILLÉE DES ERREURS")
    print("="*60)
    
    # Distribution des erreurs
    print("\n📊 Distribution des erreurs (€):")
    print(f"  Minimum     : {errors.min():>15,.0f}")
    print(f"  Q1 (25%)    : {np.percentile(errors, 25):>15,.0f}")
    print(f"  Médiane     : {np.median(errors):>15,.0f}")
    print(f"  Q3 (75%)    : {np.percentile(errors, 75):>15,.0f}")
    print(f"  Maximum     : {errors.max():>15,.0f}")
    print(f"  Écart-type  : {errors.std():>15,.0f}")
    
    # Erreurs absolues
    abs_errors = np.abs(errors)
    print("\n📊 Erreurs absolues (€):")
    print(f"  Médiane     : {np.median(abs_errors):>15,.0f}")
    print(f"  Moyenne     : {abs_errors.mean():>15,.0f}")
    print(f"  Q90         : {np.percentile(abs_errors, 90):>15,.0f}")
    print(f"  Q95         : {np.percentile(abs_errors, 95):>15,.0f}")
    
    # Pourcentage d'erreurs importantes
    print("\n📊 Analyse par seuils:")
    thresholds = [100_000, 500_000, 1_000_000, 5_000_000]
    for threshold in thresholds:
        pct = (abs_errors > threshold).mean() * 100
        print(f"  Erreurs > {threshold/1e6:.1f}M€ : {pct:>6.2f}%")
    
    # Performance par signe
    print("\n📊 Performance par signe:")
    mask_profit = y_true > 0
    mask_loss = y_true <= 0
    
    if mask_profit.sum() > 0:
        mae_profit = mean_absolute_error(y_true[mask_profit], y_pred[mask_profit])
        r2_profit = r2_score(y_true[mask_profit], y_pred[mask_profit])
        print(f"  Profits    : MAE={mae_profit:>12,.0f}€  R²={r2_profit:>6.3f}  (n={mask_profit.sum():,})")
    
    if mask_loss.sum() > 0:
        mae_loss = mean_absolute_error(y_true[mask_loss], y_pred[mask_loss])
        r2_loss = r2_score(y_true[mask_loss], y_pred[mask_loss])
        print(f"  Pertes     : MAE={mae_loss:>12,.0f}€  R²={r2_loss:>6.3f}  (n={mask_loss.sum():,})")
    
    print("="*60)

# ==================================================
## 8. VISUALISATIONS COMPLÈTES
# ==================================================

def create_comprehensive_plots(y_true, y_pred, errors, model):
    """Crée une suite complète de visualisations."""
    
    print("\n📈 Génération des visualisations...")
    
    # Configuration globale
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Dashboard principal (4 subplots)
    create_main_dashboard(y_true, y_pred, errors)
    
    # 2. Feature importance
    plot_feature_importance(model)
    
    # 3. Analyse des résidus détaillée
    plot_residuals_detailed(y_true, y_pred, errors)
    
    # 4. Distribution des prédictions vs réalité
    plot_distribution_comparison(y_true, y_pred)
    
    # 5. Analyse par quantiles
    plot_quantile_analysis(y_true, y_pred)
    
    print("✅ Toutes les visualisations ont été générées!")

def create_main_dashboard(y_true, y_pred, errors):
    """Dashboard principal avec 4 graphiques clés."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dashboard Principal - Analyse des Performances', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot principal
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=20, c='steelblue', edgecolors='none')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
    ax1.set_xlabel('Valeur Réelle (€)', fontsize=11)
    ax1.set_ylabel('Valeur Prédite (€)', fontsize=11)
    ax1.set_title('Prédictions vs Réalité', fontsize=12, fontweight='bold')
    ax1.set_xscale('symlog')
    ax1.set_yscale('symlog')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution des erreurs
    ax2 = axes[0, 1]
    ax2.hist(errors / 1e6, bins=100, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax2.set_xlabel('Erreur (M€)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Distribution des Erreurs', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Erreurs absolues par valeur réelle
    ax3 = axes[1, 0]
    abs_errors = np.abs(errors)
    ax3.scatter(np.abs(y_true), abs_errors, alpha=0.3, s=20, c='green', edgecolors='none')
    ax3.set_xlabel('|Valeur Réelle| (€)', fontsize=11)
    ax3.set_ylabel('Erreur Absolue (€)', fontsize=11)
    ax3.set_title('Erreur Absolue vs Magnitude', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Erreur relative (%)
    ax4 = axes[1, 1]
    # Filtrer les valeurs trop petites pour éviter les erreurs relatives extrêmes
    mask = np.abs(y_true) > 1000
    relative_errors = np.abs(errors[mask] / y_true[mask]) * 100
    relative_errors_clipped = np.clip(relative_errors, 0, 200)  # Cap à 200%
    ax4.hist(relative_errors_clipped, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(relative_errors_clipped.median(), color='red', linestyle='--', 
                linewidth=2, label=f'Médiane: {relative_errors_clipped.median():.1f}%')
    ax4.set_xlabel('Erreur Relative (%)', fontsize=11)
    ax4.set_ylabel('Fréquence', fontsize=11)
    ax4.set_title('Distribution des Erreurs Relatives (>1000€)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_dashboard_principal.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Dashboard principal")

def plot_feature_importance(model):
    """Feature importance améliorée."""
    try:
        importance = model.get_booster().get_score(importance_type='gain')
        df_imp = pd.DataFrame(
            list(importance.items()), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(df_imp))
        bars = plt.barh(range(len(df_imp)), df_imp['Importance'], color=colors)
        plt.yticks(range(len(df_imp)), df_imp['Feature'])
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.title('Top 20 Features - Importance des Variables', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('02_feature_importance.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance")
    except Exception as e:
        print(f"  ⚠️ Erreur feature importance: {e}")

def plot_residuals_detailed(y_true, y_pred, errors):
    """Analyse détaillée des résidus."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analyse Détaillée des Résidus', fontsize=14, fontweight='bold')
    
    # 1. Résidus vs prédictions
    ax1 = axes[0]
    ax1.scatter(y_pred, errors, alpha=0.3, s=20, c='steelblue', edgecolors='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Valeur Prédite (€)', fontsize=11)
    ax1.set_ylabel('Résidu (Réel - Prédit)', fontsize=11)
    ax1.set_title('Résidus vs Prédictions', fontsize=12)
    ax1.set_xscale('symlog')
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(errors / errors.std(), dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normalité des Résidus)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_residuals_analysis.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Analyse des résidus")

def plot_distribution_comparison(y_true, y_pred):
    """Compare les distributions des valeurs réelles vs prédites."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparaison des Distributions', fontsize=14, fontweight='bold')
    
    # 1. Distributions avec échelle log
    ax1 = axes[0]
    ax1.hist(y_true / 1e6, bins=100, alpha=0.6, label='Réel', color='blue', edgecolor='black')
    ax1.hist(y_pred / 1e6, bins=100, alpha=0.6, label='Prédit', color='red', edgecolor='black')
    ax1.set_xlabel('Résultat Net (M€)', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.set_title('Distributions Réelle vs Prédite', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots comparatifs
    ax2 = axes[1]
    data_to_plot = [y_true / 1e6, y_pred / 1e6]
    bp = ax2.boxplot(data_to_plot, labels=['Réel', 'Prédit'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Résultat Net (M€)', fontsize=11)
    ax2.set_title('Comparaison Box Plots', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('04_distribution_comparison.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Comparaison des distributions")

def plot_quantile_analysis(y_true, y_pred):
    """Analyse de performance par quantiles."""
    
    # Créer des quantiles basés sur la valeur réelle
    n_quantiles = 10
    quantiles = pd.qcut(y_true, q=n_quantiles, labels=False, duplicates='drop')
    
    # Calculer MAE et R² par quantile
    metrics_by_quantile = []
    for q in range(n_quantiles):
        mask = quantiles == q
        if mask.sum() > 10:  # Au moins 10 observations
            mae_q = mean_absolute_error(y_true[mask], y_pred[mask])
            try:
                r2_q = r2_score(y_true[mask], y_pred[mask])
            except:
                r2_q = np.nan
            mean_val = y_true[mask].mean()
            metrics_by_quantile.append({
                'quantile': q + 1,
                'mae': mae_q,
                'r2': r2_q,
                'mean_value': mean_val,
                'count': mask.sum()
            })
    
    df_quantiles = pd.DataFrame(metrics_by_quantile)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance par Quantile de Valeur', fontsize=14, fontweight='bold')
    
    # 1. MAE par quantile
    ax1 = axes[0]
    ax1.bar(df_quantiles['quantile'], df_quantiles['mae'] / 1e6, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Quantile', fontsize=11)
    ax1.set_ylabel('MAE (M€)', fontsize=11)
    ax1.set_title('MAE par Quantile', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. R² par quantile
    ax2 = axes[1]
    colors = ['red' if r2 < 0 else 'green' for r2 in df_quantiles['r2']]
    ax2.bar(df_quantiles['quantile'], df_quantiles['r2'], color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Quantile', fontsize=11)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_title('R² par Quantile', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('05_quantile_analysis.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Analyse par quantiles")

# ==================================================
## 9. RECOMMANDATIONS D'AMÉLIORATION
# ==================================================

def print_improvement_recommendations(metrics):
    """Affiche des recommandations concrètes."""
    
    print("\n" + "="*60)
    print("💡 RECOMMANDATIONS D'AMÉLIORATION")
    print("="*60)
    
    print("\n🎯 Pour améliorer le R² (actuellement {:.3f}):".format(metrics['R²']))
    print("  1. FEATURE ENGINEERING:")
    print("     • Interactions entre features (ratio_liquidite × ratio_endettement)")
    print("     • Features temporelles (tendances sur 3+ ans)")
    print("     • Agrégations par secteur (moyenne sectorielle)")
    
    print("\n  2. MODÈLE:")
    print("     • Essayer LightGBM ou CatBoost")
    print("     • Stacking (XGBoost + RandomForest)")
    print("     • Optimisation bayésienne des hyperparamètres")
    
    print("\n  3. TARGET ENCODING AVANCÉ:")
    print("     • K-fold target encoding")
    print("     • Weight of Evidence (WoE) encoding")
    print("     • Leave-one-out encoding")
    
    print("\n  4. DONNÉES:")
    print("     • Augmenter le smoothing du Target Encoder (20-30)")
    print("     • Winsorisation plus agressive (1%-99%)")
    print("     • Traiter séparément les micro/PME/grandes entreprises")
    
    print("\n📊 Pour réduire le gap CV-Test (0.66 → 0.56):")
    print("  • Régularisation plus forte (reg_alpha=1.0, reg_lambda=1.0)")
    print("  • Réduire max_depth (essayer 5 ou 4)")
    print("  • Early stopping avec eval_set sur validation")
    print("  • Calibration post-hoc des prédictions")
    
    if metrics['MAPE'] > 50:
        print("\n⚠️  MAPE élevé ({:.1f}%):".format(metrics['MAPE']))
        print("  • Modèle séparé pour petites valeurs (<50k€)")
        print("  • Transformation log sur la target (déjà fait)")
        print("  • Pondération des samples (sample_weight)")
    
    print("\n🔬 EXPÉRIMENTATIONS AVANCÉES:")
    print("  • Modèle par secteur d'activité")
    print("  • Ensembles avec vote pondéré")
    print("  • Quantile regression pour l'incertitude")
    print("  • Détection et traitement des outliers persistants")
    
    print("="*60)

# ==================================================
## 10. PIPELINE PRINCIPAL
# ==================================================

def main():
    print("="*60)
    print("🚀 PIPELINE ML - PRÉDICTION RÉSULTAT NET (Y_RN)")
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
        
        print_improvement_recommendations(metrics)
        
        print("\n✅ Pipeline terminé avec succès!")
        print("📁 Visualisations sauvegardées:")
        print("   • 01_dashboard_principal.png")
        print("   • 02_feature_importance.png")
        print("   • 03_residuals_analysis.png")
        print("   • 04_distribution_comparison.png")
        print("   • 05_quantile_analysis.png")

    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()