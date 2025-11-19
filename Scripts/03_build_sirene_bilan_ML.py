import polars as pl
import os

# --- Définition des Chemins et Paramètres ---
# Chemin du fichier EDA filtré (produit de l'étape 1)
FILE_PATH_EDA = "Data/processed/sirene_bilan_EDA.parquet" 
# Chemin du fichier final prêt pour le ML
OUTPUT_FILE_PATH_ML = "Data/processed/sirene_bilan_ML_prets.parquet" 

cible_col = "cible_HN_RésultatNet_T_plus_1"

print("Démarrage de la préparation des données pour le Machine Learning...")

# 1. Chargement du Dataset Filtré (2015-2018)
try:
    df_bilan_filtre_pl = pl.read_parquet(FILE_PATH_EDA).sort(["siren", "AnneeClotureExercice"])
except Exception as e:
    print(f"ERREUR : Impossible de charger le fichier EDA. Vérifiez le chemin : {FILE_PATH_EDA}")
    raise

# 2. Feature Engineering Avancé (T, T-1, T-2 et Deltas)
# Crée les colonnes pour toutes les observations possibles (T=2016, T=2017)
df_ml_prepa = df_bilan_filtre_pl.with_columns([
    # --- Cible Y (Résultat Net à T+1) ---
    pl.col("HN_RésultatNet").shift(-1).over("siren").alias(cible_col),

    # --- Features T-1 (N-1) ---
    pl.col("HN_RésultatNet").shift(1).over("siren").alias("ResultatNet_T_moins_1"),
    pl.col("FA_ChiffreAffairesVentes").shift(1).over("siren").alias("CA_T_moins_1"),
    
    # --- Features T-2 (N-2) ---
    pl.col("HN_RésultatNet").shift(2).over("siren").alias("ResultatNet_T_moins_2"),
    pl.col("FA_ChiffreAffairesVentes").shift(2).over("siren").alias("CA_T_moins_2"),
    
    # --- Deltas T vs T-1 (Variation 1 an) ---
    (pl.col("HN_RésultatNet") - pl.col("HN_RésultatNet").shift(1).over("siren")).alias("delta_ResultatNet_1an"),
    (pl.col("FA_ChiffreAffairesVentes") - pl.col("FA_ChiffreAffairesVentes").shift(1).over("siren")).alias("delta_CA_1an"),
    
    # --- Deltas T vs T-2 (Variation 2 ans) ---
    (pl.col("HN_RésultatNet") - pl.col("HN_RésultatNet").shift(2).over("siren")).alias("delta_ResultatNet_2ans"),
    (pl.col("FA_ChiffreAffairesVentes") - pl.col("FA_ChiffreAffairesVentes").shift(2).over("siren")).alias("delta_CA_2ans"),
])

# 3. Filtrage Final pour le ML (Aplatissement)
# On conserve uniquement les lignes qui peuvent servir d'observation complète:
# - Cible T+1 (i.e., T=2017) doit exister.
# - Historique T-2 (i.e., T=2017 nécessite 2015) doit exister.
# => On garde les observations T=2016 et T=2017.
df_ml_final = df_ml_prepa.filter(
    pl.col("AnneeClotureExercice").is_in({2016, 2017})
)
#.drop_nulls(subset=[cible_col, "ResultatNet_T_moins_2"]) # S'assure d'avoir la cible et l'historique T-2


# 4. Sauvegarde
df_ml_final.write_parquet(OUTPUT_FILE_PATH_ML)

print(f"\n✅ Fichier ML aplati créé avec {df_ml_final.shape[0]} observations.")
print(f"Périodes incluses : T=2016 (Prédit 2017) et T=2017 (Prédit 2018).")
print(f"Sauvegardé à : {OUTPUT_FILE_PATH_ML}")