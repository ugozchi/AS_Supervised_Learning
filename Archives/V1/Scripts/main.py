import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer, roc_auc_score, accuracy_score
import category_encoders as ce  # Pour le Target Encoding
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import warnings
# Supprimer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================================================
## 0. FONCTIONS UTILITAIRES ET MAPE
# ==================================================

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcule la MAPE, gère la division par zéro."""
    epsilon = 1e-8 
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def safe_divide(numerator, denominator):
    """Calcule le ratio, gère la division par zéro et les valeurs infinies."""
    denominator_safe = denominator.replace(0, np.nan)
    ratio = numerator / denominator_safe
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    return ratio

def safe_log1p(series):
    """Applique log(1 + abs(x)) * sign(x) pour gérer les valeurs positives et négatives."""
    return np.sign(series) * np.log1p(np.abs(series))


# ==================================================
## 1. DATA PREP & FEATURE ENGINEERING
# ==================================================



# Assumer que df_bilan_joined est déjà un DataFrame Pandas contenant 'departement'
df = df_full.to_pandas()

# --- Création des Features ---
df['ratio_liquidite'] = safe_divide(df['DA_TresorerieActive'], df['DL_DettesCourtTerme'])
df['ratio_stabilite_inv'] = safe_divide(1, df['anciennete_entreprise']).replace([np.inf, -np.inf], 0).fillna(0)
df['proxy_actif_taux'] = safe_divide(df['DA_TresorerieActive'] - df['DL_DettesCourtTerme'], df['anciennete_entreprise']).replace([np.inf, -np.inf], 0).fillna(0)
df['HN_RésultatNet_log'] = safe_log1p(df['HN_RésultatNet'])
df['FR_ResultatExceptionnel_log'] = safe_log1p(df['FR_ResultatExceptionnel'])
df['flag_exceptionnel'] = (df['FR_ResultatExceptionnel'] != 0).astype(int)


# 2. Nettoyage et définition des Cibles
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna() 

# --- CRÉATION DES CIBLES ---
df['target_is_profit'] = (df['cible_ResultatNet_T_plus_1'] > 0).astype(int)
df['target_magnitude_log'] = np.log1p(np.abs(df['cible_ResultatNet_T_plus_1']))


TARGET_COLUMN = 'cible_ResultatNet_T_plus_1'
COLUMNS_TO_DROP = [
    TARGET_COLUMN, 'target_is_profit', 'target_magnitude_log', 
    # Suppression des colonnes non-numériques ou leakées, MAIS ON GARDE 'departement'
    'siren', 'date_cloture_exercice', 'cible_HN_RésultatNet_T_plus_1', 
    'ID_entreprise', 'ratio_tresorerie', 'FR_ResultatExceptionnel'
]

# X = Features (maintenant incluant 'departement')
X = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
Y_reg_log = df['target_magnitude_log']
Y_cls = df['target_is_profit']


# 3. Train / test split (sur l'ensemble complet)
X_train, X_test, Y_reg_log_train, Y_reg_log_test, Y_cls_train, Y_cls_test = train_test_split(
    X, Y_reg_log, Y_cls, test_size=0.2, random_state=42
)
print(f"Shapes (Train: {X_train.shape}, Test: {X_test.shape})")

# ==================================================
## 2. ÉTAPE 1 : CLASSIFICATION DU SIGNE (PROFIT ou PERTE)
# ==================================================
print("\n[MODELE CLS] Entraînement pour prédire le SIGNE (Profit/Perte)...")

# Target Encoding du département pour la classification (basé sur Y_cls)
cls_encoder = ce.TargetEncoder(cols=['departement'])
X_train_cls_encoded = cls_encoder.fit_transform(X_train, Y_cls_train)
X_test_cls_encoded = cls_encoder.transform(X_test)


cls_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='auc', n_estimators=500,
    learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1
)
cls_model.fit(X_train_cls_encoded, Y_cls_train)
cls_pred = cls_model.predict(X_test_cls_encoded)
cls_accuracy = accuracy_score(Y_cls_test, cls_pred)

print("\n--- PERFORMANCE CLASSIFICATION (Signe) ---")
print(f"Accuracy : {cls_accuracy:.4f}")

# ==================================================
## 3. ÉTAPE 2 : RÉGRESSION DE LA MAGNITUDE (CV=5)
# ==================================================
print("\n[MODELE REG] Cross-Validation (CV=5) avec TARGET ENCODING sur le Département...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores_cv = []
r2_scores_cv = []
mape_scores_cv = []

# Modèle de régression de base (paramètres fixés sans RandomizedSearch)
base_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=1000, max_depth=7, 
    learning_rate=0.05, random_state=42, n_jobs=-1
)

# Boucle de Cross-Validation (KFold est appliqué sur l'ensemble X et Y complets)
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    
    # Séparation des données du fold
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    Y_reg_log_train_fold = Y_reg_log.iloc[train_index]
    
    # 1. TARGET ENCODING (Calculé uniquement sur le TRAIN du fold)
    reg_encoder = ce.TargetEncoder(cols=['departement']) 
    
    X_train_encoded = reg_encoder.fit_transform(X_train_fold, Y_reg_log_train_fold)
    X_val_encoded = reg_encoder.transform(X_val_fold)
    
    # 2. Entraînement et Prédiction
    fold_reg_model = base_regressor
    fold_reg_model.fit(X_train_encoded, Y_reg_log_train_fold)
    
    pred_log_mag = fold_reg_model.predict(X_val_encoded)
    pred_magnitude = np.expm1(pred_log_mag)
    
    # 3. Évaluation sur le Y original (Magnitude Absolue)
    y_val_original = df.loc[Y_reg_log.iloc[val_index].index, TARGET_COLUMN]

    mae_scores_cv.append(mean_absolute_error(y_val_original.abs(), pred_magnitude))
    r2_scores_cv.append(r2_score(y_val_original.abs(), pred_magnitude))
    mape_scores_cv.append(mean_absolute_percentage_error(y_val_original.abs(), pred_magnitude))

    print(f"Fold {fold+1}: MAE={mae_scores_cv[-1]:,.0f}€, R2={r2_scores_cv[-1]:.4f}")


# 4. Entraînement du modèle final sur l'ensemble TRAIN complet (pour le Test Set)
final_reg_encoder = ce.TargetEncoder(cols=['departement']) 
X_train_final_encoded = final_reg_encoder.fit_transform(X_train, Y_reg_log_train)

reg_model_final = base_regressor
reg_model_final.fit(X_train_final_encoded, Y_reg_log_train)


print("\n--- PERFORMANCE MOYENNE CROSS-VALIDATION (Magnitude) ---")
print(f"MAE (CV) : {np.mean(mae_scores_cv):,.2f} €")
print(f"R² (CV)  : {np.mean(r2_scores_cv):.4f}")
# print(f"MAPE (CV): {np.mean(mape_scores_cv):.2f} %")


# ==================================================
## 4. PRÉDICTION FINALE & ÉVALUATION GLOBALE (Test Set)
# ==================================================

# 1. Application de l'encodage final sur le Test Set
X_test_final_encoded = final_reg_encoder.transform(X_test) 

# 2. Prédiction de la Magnitude (Log-échelle)
predictions_magnitude_log = reg_model_final.predict(X_test_final_encoded)
predictions_magnitude = np.expm1(predictions_magnitude_log)
predictions_magnitude[predictions_magnitude < 0] = 0

# 3. Prédiction du Signe (utilisant le cls_pred calculé sur X_test_cls_encoded)
predictions_signe = np.where(cls_pred == 1, 1, -1)
final_predictions = predictions_signe * predictions_magnitude

# 4. Évaluation
y_test_original = df.loc[Y_cls_test.index, TARGET_COLUMN] 

final_mae = mean_absolute_error(y_test_original, final_predictions)
final_r2 = r2_score(y_test_original, final_predictions)
final_mape = mean_absolute_percentage_error(y_test_original, final_predictions)

print("\n--- PERFORMANCE GLOBALE FINALE (Test Set) ---")
print(f"MAE : {final_mae:,.2f} € ")
print(f"R²  : {final_r2:.4f} ")
# print(f"MAPE: {final_mape:.2f} %")


# ==================================================
## 5. PLOTS ET ANALYSE D'IMPORTANCE
# ==================================================

# 5.1 FEATURE IMPORTANCE (Modèle de REGRESSION de Magnitude)
print("\n[PLOTS] Génération de la Feature Importance pour le modèle de Magnitude...")

# La colonne 'departement' sera affichée comme sa valeur encodée
importance = reg_model_final.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(20), 
    palette="viridis"
)
plt.title("Variables les plus influentes sur la MAGNITUDE (incl. Target Encoding)")
plt.tight_layout()
plt.savefig("feature_importance_target_encoded.png")

# 5.2 PLOT PRÉDICTION vs RÉALITÉ
plt.figure(figsize=(10, 10))
plt.scatter(y_test_original, final_predictions, alpha=0.4, s=10)
p1 = max(max(final_predictions), max(y_test_original))
p0 = min(min(final_predictions), min(y_test_original))
plt.plot([p0, p1], [p0, p1], 'r--')
plt.xlabel('Vrai Résultat Net N+1')
plt.ylabel('Résultat Prédit N+1')
plt.title('Précision du Modèle Hybride (Target Encoding)')
plt.xscale('symlog')
plt.yscale('symlog')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("prediction_vs_realite_target_encoded.png")

print("\nGraphiques générés : 'feature_importance_target_encoded.png' et 'prediction_vs_realite_target_encoded.png'")