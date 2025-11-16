import polars as pl
import pyarrow.parquet as pq
import sys
import os

# --- 1. GESTION DES CHEMINS (ROBUSTE) ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.join(os.getcwd(), "Scripts")

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

PATH_INPI = os.path.join(PROJECT_ROOT, "Data/raw/ExportDetailBilan.parquet")
PATH_OUTPUT = os.path.join(PROJECT_ROOT, "Data/processed/sirene_bilan.parquet") # Cible du Makefile

print("--- Lancement Script 02: Création du 'DNA Financier' (Pivot Expert) ---")

# --- 2. VÉRIFICATION DES FICHIERS ---
if not os.path.exists(PATH_INPI):
    print(f"ERREUR FATALE: Fichier brut manquant : {PATH_INPI}", file=sys.stderr)
    print("Assure-toi que les fichiers sont dans 'Data/raw/' (as-tu lancé 'make download'?)", file=sys.stderr)
    sys.exit(1)

# --- 3. LISTE DES CODES "DIAMANT" ---
CODES_A_GARDER = [
    'HN', 'FA', 'FB', 'CJ-CK', 'DL', 'DM', 'DA', 'FJ', 'FR', 'DF', 'EG'
]
RENAMING_MAP = {
    'HN': 'HN_RésultatNet', 'FA': 'FA_ChiffreAffairesVentes', 'FB': 'FB_AchatsMarchandises',
    'CJ-CK': 'CJCK_TotalActifBrut', 'DL': 'DL_DettesCourtTerme', 'DM': 'DM_DettesLongTerme',
    'DA': 'DA_TresorerieActive', 'FJ': 'FJ_ResultatFinancier', 'FR': 'FR_ResultatExceptionnel',
    'DF': 'DF_CapitauxPropres', 'EG': 'EG_ImpotsTaxes'
}

# --- 4. LECTURE (avec le bypass pyarrow) ---
print(f"Lecture du fichier INPI (via PyArrow) : {PATH_INPI}...")
try:
    table_inpi = pq.read_table(PATH_INPI, columns=["siren", "liasse", "date_cloture_exercice"])
    df_bilan_brut = pl.from_arrow(table_inpi)
except Exception as e:
    print(f"ERREUR: La lecture de {PATH_INPI} a échoué (même avec le bypass).", file=sys.stderr)
    print(f"Détail: {e}", file=sys.stderr)
    sys.exit(1)

# --- 5. TRANSFORMATION (Explode + Séparation) ---
print("Transformation (explode) de la liasse...")
df_flat = df_bilan_brut.explode("liasse")
df_struct = df_flat.select(
    "siren",
    "date_cloture_exercice",
    pl.col("liasse").struct.field("field_0").alias("code_bilan"),
    pl.col("liasse").struct.field("field_1").alias("valeur")
)

# --- 6. LE FILTRE "EXPERT" ---
print(f"Filtrage: on ne garde que les {len(CODES_A_GARDER)} codes 'diamant'...")
df_filtered = df_struct.filter(
    pl.col("code_bilan").is_in(CODES_A_GARDER)
)

# --- 7. Le PIVOT (rapide) ---
print("PIVOT 'Expert'...")
df_wide = df_filtered.pivot(
    index=["siren", "date_cloture_exercice"],
    columns="code_bilan",
    values="valeur"
).fill_null(0)

# --- 8. Renommage (pour les noms propres) ---
df_wide = df_wide.rename(RENAMING_MAP)

# --- 9. LE FEATURE ENGINEERING "EXPERT" (Ratios) ---
print("Création des 7 ratios 'experts'...")
df_dna_expert = df_wide.with_columns(
    pl.col("date_cloture_exercice").dt.year().alias("AnneeClotureExercice"),
    (pl.col("HN_RésultatNet") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_rentabilite_nette"),
    ((pl.col("DL_DettesCourtTerme") + pl.col("DM_DettesLongTerme")) / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_endettement"),
    ((pl.col("FA_ChiffreAffairesVentes") - pl.col("FB_AchatsMarchandises")) / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_marge_brute"),
    (pl.col("DF_CapitauxPropres") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_capitaux_propres"),
    (pl.col("DA_TresorerieActive") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_tresorerie"),
    (pl.col("FJ_ResultatFinancier") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_financier"),
    (pl.col("FR_ResultatExceptionnel") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_exceptionnel")
)

# --- 10. SAUVEGARDE ! ---
print(f"Sauvegarde du Master File INPI dans {PATH_OUTPUT}...")
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True) # S'assure que le dossier existe
df_dna_expert.write_parquet(PATH_OUTPUT)
print(f"--- Script 02 (Master File INPI) Terminé avec Succès ---")