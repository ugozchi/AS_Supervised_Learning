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
## 0. CONFIGURATION
# ==================================================

DATA_PATH = 'Data/processed/sirene_final2.parquet'
TARGET_RN_NPLUS1 = 'Y_RN' 
RANDOM_SEED = 42

TARGET_ENCODING_COLS = ['departement', 'secteur_NAF_2chiffres']

FINAL_FEATURE_WHITELIST = [
    # Features baseline
    'ratio_rentabilite_nette', 'ratio_endettement', 'ratio_tresorerie', 
    'ratio_resultat_financier', 'ratio_resultat_exceptionnel', 
    'ratio_liquidite', 'ratio_stabilite_inv', 'proxy_actif_taux',
    'HN_RésultatNet_log', 'FR_ResultatExceptionnel_log', 
    'DL_DettesCourtTerme_log', 'CJCK_TotalActifBrut_log',
    'ratio_dette_ct_vs_actif', 'ratio_tresorerie_vs_dette_ct', 
    'flag_exceptionnel', 
    'taux_croissance_RN_N-1', 'taux_croissance_RN_N-2', 
    'departement', 'secteur_NAF_2chiffres',
    # 🆕 Nouvelles features spécifiques pertes
    'flag_dettes_explosives',
    'flag_perte_consecutive',
    'ratio_solvabilite_immediat',
]

COLS_TO_WINSORIZE = [
    TARGET_RN_NPLUS1, 'HN_RésultatNet', 'CJCK_TotalActifBrut', 
    'DL_DettesCourtTerme', 'FR_ResultatExceptionnel'
]
WINSOR_LOW = 0.025
WINSOR_HIGH = 0.975


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERREUR: {file_path}")
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
## 1. FEATURE ENGINEERING OPTIMISÉ
# ==================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering avec nettoyage des ratios aberrants."""
    
    print("\n🔧 Winsorisation (2.5%-97.5%)...")
    for col in COLS_TO_WINSORIZE:
        if col in df.columns:
            lower_bound = df[col].quantile(WINSOR_LOW)
            upper_bound = df[col].quantile(WINSOR_HIGH)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # --- FEATURES BASELINE ---
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
    
    # 🔥 NETTOYAGE DES RATIOS ABERRANTS
    print("🧹 Nettoyage des ratios aberrants...")
    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    for col in ratio_cols:
        if col in df.columns:
            # Clipper les valeurs extrêmes
            df[col] = df[col].clip(-10, 10)
            # Remplacer NaN par 0
            df[col] = df[col].fillna(0)
    
    # 🆕 NOUVELLES FEATURES SPÉCIFIQUES AUX PERTES
    print("🆕 Ajout de features spécifiques pertes...")
    
    # 1. Flag dettes explosives
    df['flag_dettes_explosives'] = (df['ratio_endettement'] > 2.0).astype(int)
    
    # 2. Flag perte consécutive (2 années de suite)
    df['flag_perte_consecutive'] = (
        (df['variation_resultat_net_N-1'] < 0) & 
        (df['variation_resultat_net_N-2'] < 0)
    ).astype(int)
    
    # 3. Ratio solvabilité immédiat
    df['ratio_solvabilite_immediat'] = safe_divide(
        df['DA_TresorerieActive'], 
        df['DL_DettesCourtTerme'] + 1
    )
    
    # --- TARGETS ---
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
    
    nb_features = len([c for c in df.columns if c in FINAL_FEATURE_WHITELIST])
    print(f"✅ Features: {nb_features} | Lignes: {len(df):,}")
    return df

# ==================================================
## 2. SPLIT TEMPOREL
# ==================================================

def split_data(df: pd.DataFrame) -> Tuple:
    df_sorted = df.sort_values(by='AnneeClotureExercice').reset_index(drop=True)
    split_point = int(len(df_sorted) * 0.8)
    
    cols_to_remove = [TARGET_RN_NPLUS1, 'target_is_profit', 'target_magnitude_log', 'AnneeClotureExercice']
    X = df_sorted.drop(columns=cols_to_remove, errors='ignore')
    
    Y_reg_log = df_sorted['target_magnitude_log']
    Y_cls = df_sorted['target_is_profit']
    Y_raw = df_sorted[TARGET_RN_NPLUS1]
    
    X_train = X.iloc[:split_point].copy()
    X_test = X.iloc[split_point:].copy()
    Y_reg_log_train = Y_reg_log.iloc[:split_point].copy()
    Y_cls_train = Y_cls.iloc[:split_point].copy()
    Y_cls_test = Y_cls.iloc[split_point:].copy()
    Y_raw_train = Y_raw.iloc[:split_point].copy()
    Y_raw_test = Y_raw.iloc[split_point:].copy()

    print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, Y_reg_log_train, Y_cls_train, Y_cls_test, Y_raw_train, Y_raw_test

# ==================================================
## 3. CLASSIFICATION - BASELINE
# ==================================================

def train_and_evaluate_cls(X_train, X_test, Y_train, Y_test):
    cls_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
    X_train_encoded = cls_encoder.fit_transform(X_train, Y_train)
    X_test_encoded = cls_encoder.transform(X_test)

    print("\n🔵 [CLASSIFICATION] Prédiction du signe...")
    cls_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='auc',
        n_estimators=700, 
        learning_rate=0.03,
        max_depth=7,
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    cls_model.fit(X_train_encoded, Y_train)

    cls_pred = cls_model.predict(X_test_encoded)
    cls_accuracy = accuracy_score(Y_test, cls_pred)
    print(f"✅ Accuracy: {cls_accuracy:.4f}")
    
    return cls_model, cls_encoder

# ==================================================
## 4. RÉGRESSION SÉPARÉE PERTES/PROFITS ⭐
# ==================================================

def train_regression_models_separated(X_train, Y_reg_log_train, Y_raw_train):
    """
    🔥 INNOVATION MAJEURE : Entraîne 2 modèles séparés.
    - Modèle A : seulement sur PROFITS
    - Modèle B : seulement sur PERTES
    """
    
    print("\n🟢 [RÉGRESSION SÉPARÉE] Entraînement de 2 modèles distincts...")
    
    # Séparer train en profits/pertes
    mask_profits = Y_raw_train > 0
    mask_pertes = Y_raw_train <= 0
    
    print(f"  📊 Profits: {mask_profits.sum():,} samples")
    print(f"  📊 Pertes:  {mask_pertes.sum():,} samples")
    
    # --- MODÈLE PROFITS ---
    print("\n  🟢 Modèle PROFITS (CV=3)...")
    X_train_profits = X_train[mask_profits].copy()
    Y_train_profits = Y_reg_log_train[mask_profits].copy()
    
    model_profits, encoder_profits = train_single_regression_model(
        X_train_profits, Y_train_profits, 
        model_name="Profits", n_folds=3
    )
    
    # --- MODÈLE PERTES ---
    print("\n  🔴 Modèle PERTES (CV=3)...")
    X_train_pertes = X_train[mask_pertes].copy()
    Y_train_pertes = Y_reg_log_train[mask_pertes].copy()
    
    model_pertes, encoder_pertes = train_single_regression_model(
        X_train_pertes, Y_train_pertes,
        model_name="Pertes", n_folds=3
    )
    
    return {
        'profits': model_profits,
        'pertes': model_pertes,
        'encoder_profits': encoder_profits,
        'encoder_pertes': encoder_pertes
    }

def train_single_regression_model(X_train, Y_train, model_name="Model", n_folds=3):
    """Entraîne un seul modèle de régression avec CV."""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    r2_scores = []
    
    # 🔥 RÉGULARISATION ADAPTÉE selon le nombre de données
    is_small_dataset = len(X_train) < 50000
    
    if is_small_dataset:  # Pour PERTES (peu de données)
        base_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,      # 🔥 Réduit de 1500 → 500
            'max_depth': 3,           # 🔥 Réduit de 6 → 3
            'learning_rate': 0.05,    # 🔥 Augmenté (moins d'overfitting)
            'subsample': 0.6,         # 🔥 Réduit
            'colsample_bytree': 0.6,  # 🔥 Réduit
            'reg_alpha': 2.0,         # 🔥 x4 régularisation L1
            'reg_lambda': 2.0,        # 🔥 x4 régularisation L2
            'min_child_weight': 5,    # 🔥 Nouveau
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
        }
    else:  # Pour PROFITS (beaucoup de données)
        base_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
        }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        Y_tr = Y_train.iloc[train_idx]
        Y_val = Y_train.iloc[val_idx]
        
        fold_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
        X_tr_enc = fold_encoder.fit_transform(X_tr, Y_tr)
        X_val_enc = fold_encoder.transform(X_val)
        
        fold_model = xgb.XGBRegressor(**base_params)
        fold_model.fit(X_tr_enc, Y_tr)
        
        pred = fold_model.predict(X_val_enc)
        r2 = r2_score(Y_val, pred)
        r2_scores.append(r2)
        print(f"    Fold {fold}: R² = {r2:.4f}")
    
    print(f"    → R² moyen: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    
    # Entraînement final
    final_encoder = ce.TargetEncoder(cols=TARGET_ENCODING_COLS, smoothing=10)
    X_train_enc = final_encoder.fit_transform(X_train, Y_train)
    
    final_model = xgb.XGBRegressor(**base_params)
    final_model.fit(X_train_enc, Y_train)
    
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
## 6. ÉVALUATION AVEC MODÈLES SÉPARÉS
# ==================================================

def evaluate_separated_models(cls_model, cls_encoder, reg_models, 
                               X_test, Y_cls_test, Y_raw_test):
    """Évaluation en utilisant le bon modèle selon le signe prédit."""
    
    print("\n🎯 [ÉVALUATION] Prédictions avec modèles séparés...")
    
    # Encodage pour classification
    X_test_cls_enc = cls_encoder.transform(X_test)
    
    # Prédiction du signe
    pred_sign = cls_model.predict(X_test_cls_enc)
    pred_sign_binary = np.where(pred_sign == 1, 1, -1)
    
    mask_pred_profits = (pred_sign == 1)
    mask_pred_pertes = (pred_sign == 0)
    
    print(f"  📊 Prédictions profits: {mask_pred_profits.sum():,}")
    print(f"  📊 Prédictions pertes:  {mask_pred_pertes.sum():,}")
    
    # Prédire avec le bon modèle
    pred_magnitude_log = np.zeros(len(X_test))
    
    # Modèle PROFITS
    if mask_pred_profits.sum() > 0:
        X_test_profits = X_test[mask_pred_profits].copy()
        X_test_profits_enc = reg_models['encoder_profits'].transform(X_test_profits)
        pred_magnitude_log[mask_pred_profits] = reg_models['profits'].predict(X_test_profits_enc)
    
    # Modèle PERTES
    if mask_pred_pertes.sum() > 0:
        X_test_pertes = X_test[mask_pred_pertes].copy()
        X_test_pertes_enc = reg_models['encoder_pertes'].transform(X_test_pertes)
        pred_magnitude_log[mask_pred_pertes] = reg_models['pertes'].predict(X_test_pertes_enc)
    
    # Conversion magnitude
    pred_magnitude = np.expm1(pred_magnitude_log)
    pred_magnitude = np.maximum(pred_magnitude, 0)
    
    # Combinaison finale
    final_pred = pred_sign_binary * pred_magnitude
    
    # Métriques globales
    metrics = calculate_metrics(Y_raw_test, final_pred)
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE FINALE - MODÈLES SÉPARÉS")
    print("="*60)
    print(f"MAE       : {metrics['MAE']:>15,.2f} €")
    print(f"MedAE     : {metrics['MedAE']:>15,.2f} € (médiane)")
    print(f"RMSE      : {metrics['RMSE']:>15,.2f} €")
    print(f"MAPE      : {metrics['MAPE']:>15.2f} %")
    print(f"R²        : {metrics['R²']:>15.4f}")
    print("="*60)
    
    # Analyse par signe
    errors = Y_raw_test - final_pred
    analyze_errors_separated(Y_raw_test, final_pred, errors)
    
    # Visualisations
    create_plots(Y_raw_test, final_pred, errors)
    
    return final_pred, metrics

# ==================================================
## 7. ANALYSE DES ERREURS
# ==================================================

def analyze_errors_separated(y_true, y_pred, errors):
    print("\n" + "="*60)
    print("🔍 ANALYSE DÉTAILLÉE")
    print("="*60)
    
    abs_errors = np.abs(errors)
    print(f"\n📊 Erreurs globales:")
    print(f"  Médiane : {np.median(abs_errors):>12,.0f} €")
    print(f"  Moyenne : {abs_errors.mean():>12,.0f} €")
    print(f"  Q95     : {np.percentile(abs_errors, 95):>12,.0f} €")
    
    print("\n📊 Performance par signe:")
    for label, mask in [('Profits', y_true > 0), ('Pertes', y_true <= 0)]:
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            r2 = r2_score(y_true[mask], y_pred[mask])
            mape = calculate_mape(y_true[mask], y_pred[mask])
            print(f"  {label:>8}: MAE={mae:>10,.0f}€ | R²={r2:>6.3f} | MAPE={mape:>6.1f}% | n={mask.sum():,}")
    
    print("="*60)

def create_plots(y_true, y_pred, errors):
    print("\n📈 Génération des visualisations...")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modèle Optimisé - Régression Séparée Pertes/Profits', fontsize=16, fontweight='bold')
    
    # Scatter
    ax1 = axes[0, 0]
    mask_profit = y_true > 0
    ax1.scatter(y_true[mask_profit], y_pred[mask_profit], alpha=0.3, s=15, c='green', label='Profits')
    ax1.scatter(y_true[~mask_profit], y_pred[~mask_profit], alpha=0.3, s=15, c='red', label='Pertes')
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lim, lim, 'k--', lw=2)
    ax1.set_xlabel('Réel (€)')
    ax1.set_ylabel('Prédit (€)')
    ax1.set_title('Prédictions vs Réalité')
    ax1.set_xscale('symlog')
    ax1.set_yscale('symlog')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution erreurs
    ax2 = axes[0, 1]
    ax2.hist(errors / 1e6, bins=100, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Erreur (M€)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Erreurs')
    ax2.grid(True, alpha=0.3)
    
    # Erreur absolue
    ax3 = axes[1, 0]
    abs_errors = np.abs(errors)
    ax3.scatter(np.abs(y_true), abs_errors, alpha=0.3, s=15, c='steelblue')
    ax3.set_xlabel('|Réel| (€)')
    ax3.set_ylabel('Erreur Absolue (€)')
    ax3.set_title('Erreur vs Magnitude')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Box plot par signe
    ax4 = axes[1, 1]
    data_box = [
        errors[y_true > 0] / 1e6,
        errors[y_true <= 0] / 1e6
    ]
    bp = ax4.boxplot(data_box, tick_labels=['Profits', 'Pertes'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.axhline(0, color='red', linestyle='--', lw=1)
    ax4.set_ylabel('Erreur (M€)')
    ax4.set_title('Distribution Erreurs par Signe')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('dashboard_optimized.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ dashboard_optimized.png")

# ==================================================
## 8. COMPARAISON BASELINE
# ==================================================

def print_comparison(metrics, baseline):
    print("\n" + "="*60)
    print("📊 COMPARAISON BASELINE → OPTIMISÉ")
    print("="*60)
    
    improvements = []
    
    for metric in ['MAE', 'R²', 'MAPE']:
        old = baseline.get(metric, 0)
        new = metrics.get(metric, 0)
        
        if metric == 'MAE':
            delta = ((new - old) / old * 100) if old != 0 else 0
            symbol = "✅" if delta < -2 else "⚠️" if delta < 0 else "❌"
            improvements.append(delta < 0)
            print(f"  {symbol} {metric:>6}: {old:>10,.0f}€ → {new:>10,.0f}€  ({delta:+.1f}%)")
        elif metric == 'MAPE':
            delta = new - old
            symbol = "✅" if delta < -5 else "⚠️" if delta < 0 else "❌"
            improvements.append(delta < 0)
            print(f"  {symbol} {metric:>6}: {old:>10.1f}% → {new:>10.1f}%  ({delta:+.1f} pts)")
        else:  # R²
            delta = new - old
            symbol = "✅" if delta > 0.02 else "⚠️" if delta > 0 else "❌"
            improvements.append(delta > 0)
            print(f"  {symbol} {metric:>6}: {old:>10.4f} → {new:>10.4f}  ({delta:+.4f})")
    
    print("\n🎯 Améliorations:")
    print(f"  • Ratios aberrants nettoyés")
    print(f"  • 3 nouvelles features spécifiques pertes")
    print(f"  • 2 modèles séparés (profits/pertes)")
    print(f"  • {sum(improvements)}/3 métriques améliorées")
    
    print("="*60)

# ==================================================
## 9. PIPELINE PRINCIPAL
# ==================================================

def main():
    print("="*60)
    print("🚀 PIPELINE OPTIMISÉ - MODÈLES SÉPARÉS")
    print("="*60)
    
    try:
        df_raw = load_data(DATA_PATH)
        df_processed = feature_engineering(df_raw)
        
        (X_train, X_test, Y_reg_log_train, Y_cls_train, 
         Y_cls_test, Y_raw_train, Y_raw_test) = split_data(df_processed)
        
        # Classification
        cls_model, cls_encoder = train_and_evaluate_cls(
            X_train, X_test, Y_cls_train, Y_cls_test
        )
        
        # Régression SÉPARÉE
        reg_models = train_regression_models_separated(
            X_train, Y_reg_log_train, Y_raw_train
        )
        
        # Évaluation
        final_pred, metrics = evaluate_separated_models(
            cls_model, cls_encoder, reg_models,
            X_test, Y_cls_test, Y_raw_test
        )
        
        # Comparaison
        baseline = {'MAE': 352099, 'R²': 0.5644, 'MAPE': 181.58}
        print_comparison(metrics, baseline)
        
        print("\n✅ Pipeline optimisé terminé!")
        print("📁 Visualisation: dashboard_optimized.png")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()