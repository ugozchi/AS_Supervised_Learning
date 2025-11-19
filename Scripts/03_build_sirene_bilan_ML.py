import polars as pl

# --- Définition des Chemins et Variables ---
FILE_PATH_EDA = "../Data/processed/sirene_bilan_EDA.parquet" 
OUTPUT_FILE_PATH_ML = "Data/processed/sirene_bilan_ML_prets.parquet" 
cible_col = "cible_HN_RésultatNet_T_plus_1"

# 1. Chargement et Feature Engineering Initial (pour avoir Y, T-1)
try:
    df_bilan_filtre_pl = pl.read_parquet(FILE_PATH_EDA).sort(["siren", "AnneeClotureExercice"])
except Exception as e:
    print(f"ERREUR : Impossible de charger le fichier EDA. Vérifiez le chemin : {FILE_PATH_EDA}")
    raise

# 2. Enrichissement (T, T-1, T-2 et Deltas)
df_ml_prepa = df_bilan_filtre_pl.with_columns([
    # --- Cible Y ---
    pl.col("HN_RésultatNet").shift(-1).over("siren").alias(cible_col),

    # --- Features T-1 ---
    pl.col("HN_RésultatNet").shift(1).over("siren").alias("ResultatNet_T_moins_1"),
    pl.col("FA_ChiffreAffairesVentes").shift(1).over("siren").alias("CA_T_moins_1"),
    
    # --- Features T-2 ---
    pl.col("HN_RésultatNet").shift(2).over("siren").alias("ResultatNet_T_moins_2"),
    pl.col("FA_ChiffreAffairesVentes").shift(2).over("siren").alias("CA_T_moins_2"),
    
    # --- Deltas (Variation T vs T-1) ---
    (pl.col("HN_RésultatNet") - pl.col("HN_RésultatNet").shift(1).over("siren")).alias("delta_ResultatNet_1an"),
    (pl.col("FA_ChiffreAffairesVentes") - pl.col("FA_ChiffreAffairesVentes").shift(1).over("siren")).alias("delta_CA_1an"),
    
    # --- Deltas (Variation T vs T-2) ---
    (pl.col("HN_RésultatNet") - pl.col("HN_RésultatNet").shift(2).over("siren")).alias("delta_ResultatNet_2ans"),
    (pl.col("FA_ChiffreAffairesVentes") - pl.col("FA_ChiffreAffairesVentes").shift(2).over("siren")).alias("delta_CA_2ans"),
])

# 3. Filtrage : On garde uniquement les années où T-2 existe (T >= 2017) et où la Cible existe (T <= 2017)
# => On garde T=2017 (Features 2017, Cible 2018) et T=2016 (Features 2016, Cible 2017)
# On filtre les lignes où toutes les features T-2 et la cible existent.
df_ml_final = df_ml_prepa.filter(
    pl.col("AnneeClotureExercice").is_in({2016, 2017})
).drop_nulls(subset=[cible_col, "ResultatNet_T_moins_2"])


# 4. Sauvegarde du fichier ML aplati
df_ml_final.write_parquet(OUTPUT_FILE_PATH_ML)

print(f"✅ Fichier ML aplati (contient 2016 -> 2017 et 2017 -> 2018) créé à : {OUTPUT_FILE_PATH_ML}")