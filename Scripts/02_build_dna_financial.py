# import polars as pl
# import pyarrow.parquet as pq
# import sys
# import os

# # --- 1. GESTION DES CHEMINS (ROBUSTE) ---
# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     SCRIPT_DIR = os.path.join(os.getcwd(), "Scripts")

# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# PATH_INPI = os.path.join(PROJECT_ROOT, "Data/raw/ExportDetailBilan.parquet")
# PATH_OUTPUT = os.path.join(PROJECT_ROOT, "Data/processed/sirene_bilan.parquet") # Cible du Makefile

# print("--- Lancement Script 02: Création du 'DNA Financier' (Pivot Expert) ---")

# # --- 2. VÉRIFICATION DES FICHIERS ---
# if not os.path.exists(PATH_INPI):
#     print(f"ERREUR FATALE: Fichier brut manquant : {PATH_INPI}", file=sys.stderr)
#     print("Assure-toi que les fichiers sont dans 'Data/raw/' (as-tu lancé 'make download'?)", file=sys.stderr)
#     sys.exit(1)

# # --- 3. LISTE DES CODES "DIAMANT" ---
# CODES_A_GARDER = [
#     'HN', 'FA', 'FB', 'CJ-CK', 'DL', 'DM', 'DA', 'FJ', 'FR', 'DF', 'EG'
# ]
# RENAMING_MAP = {
#     'HN': 'HN_RésultatNet', 'FA': 'FA_ChiffreAffairesVentes', 'FB': 'FB_AchatsMarchandises',
#     'CJ-CK': 'CJCK_TotalActifBrut', 'DL': 'DL_DettesCourtTerme', 'DM': 'DM_DettesLongTerme',
#     'DA': 'DA_TresorerieActive', 'FJ': 'FJ_ResultatFinancier', 'FR': 'FR_ResultatExceptionnel',
#     'DF': 'DF_CapitauxPropres', 'EG': 'EG_ImpotsTaxes'
# }

# # --- 4. LECTURE (avec le bypass pyarrow) ---
# print(f"Lecture du fichier INPI (via PyArrow) : {PATH_INPI}...")
# try:
#     table_inpi = pq.read_table(PATH_INPI, columns=["siren", "liasse", "date_cloture_exercice"])
#     df_bilan_brut = pl.from_arrow(table_inpi)
# except Exception as e:
#     print(f"ERREUR: La lecture de {PATH_INPI} a échoué (même avec le bypass).", file=sys.stderr)
#     print(f"Détail: {e}", file=sys.stderr)
#     sys.exit(1)

# # --- 5. TRANSFORMATION (Explode + Séparation) ---
# print("Transformation (explode) de la liasse...")
# df_flat = df_bilan_brut.explode("liasse")
# df_struct = df_flat.select(
#     "siren",
#     "date_cloture_exercice",
#     pl.col("liasse").struct.field("field_0").alias("code_bilan"),
#     pl.col("liasse").struct.field("field_1").alias("valeur")
# )

# # --- 6. LE FILTRE "EXPERT" ---
# print(f"Filtrage: on ne garde que les {len(CODES_A_GARDER)} codes 'diamant'...")
# df_filtered = df_struct.filter(
#     pl.col("code_bilan").is_in(CODES_A_GARDER)
# )

# # --- 7. Le PIVOT (rapide) ---
# print("PIVOT 'Expert'...")
# df_wide = df_filtered.pivot(
#     index=["siren", "date_cloture_exercice"],
#     columns="code_bilan",
#     values="valeur"
# ).fill_null(0)

# # --- 8. Renommage (pour les noms propres) ---
# df_wide = df_wide.rename(RENAMING_MAP)

# # --- 9. LE FEATURE ENGINEERING "EXPERT" (Ratios) ---
# print("Création des 7 ratios 'experts'...")
# df_dna_expert = df_wide.with_columns(
#     pl.col("date_cloture_exercice").dt.year().alias("AnneeClotureExercice"),
#     (pl.col("HN_RésultatNet") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_rentabilite_nette"),
#     ((pl.col("DL_DettesCourtTerme") + pl.col("DM_DettesLongTerme")) / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_endettement"),
#     ((pl.col("FA_ChiffreAffairesVentes") - pl.col("FB_AchatsMarchandises")) / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_marge_brute"),
#     (pl.col("DF_CapitauxPropres") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_capitaux_propres"),
#     (pl.col("DA_TresorerieActive") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_tresorerie"),
#     (pl.col("FJ_ResultatFinancier") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_financier"),
#     (pl.col("FR_ResultatExceptionnel") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_exceptionnel")
# )

# # --- 10. SAUVEGARDE ! ---
# print(f"Sauvegarde du Master File INPI dans {PATH_OUTPUT}...")
# os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True) # S'assure que le dossier existe
# df_dna_expert.write_parquet(PATH_OUTPUT)
# print(f"--- Script 02 (Master File INPI) Terminé avec Succès ---")

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
    print(f"DEBUG: {len(df_bilan_brut)} lignes lues")
except Exception as e:
    print(f"ERREUR: La lecture de {PATH_INPI} a échoué (même avec le bypass).", file=sys.stderr)
    print(f"Détail: {e}", file=sys.stderr)
    sys.exit(1)

# --- DEBUG: Inspecter la structure de la colonne 'liasse' ---
print("DEBUG: Inspection du schéma de la colonne 'liasse'...")
print(f"Type de la colonne 'liasse': {df_bilan_brut['liasse'].dtype}")

# Prendre un échantillon pour voir la structure
sample = df_bilan_brut.select("liasse").head(1)
print(f"DEBUG: Premier élément de 'liasse': {sample['liasse'][0]}")

# Si c'est une structure, afficher les noms des champs
if isinstance(df_bilan_brut['liasse'].dtype, pl.List):
    print("DEBUG: 'liasse' est une List")
    # Exploser pour voir le contenu
    sample_exploded = df_bilan_brut.head(1).explode("liasse")
    print(f"DEBUG: Type après explode: {sample_exploded['liasse'].dtype}")
    print(f"DEBUG: Premier élément après explode: {sample_exploded['liasse'][0]}")
    
    # Si c'est un Struct, afficher les noms des champs
    if isinstance(sample_exploded['liasse'].dtype, pl.Struct):
        field_names = sample_exploded['liasse'].dtype.fields
        print(f"DEBUG: Noms des champs dans le Struct: {[f.name for f in field_names]}")

# --- 5. TRANSFORMATION (Explode + Séparation) ---
print("Transformation (explode) de la liasse...")
df_flat = df_bilan_brut.explode("liasse")

# Inspecter la structure après explode
print(f"DEBUG: Type de 'liasse' après explode: {df_flat['liasse'].dtype}")

# Vérifier si c'est un Struct et obtenir les noms de champs
if isinstance(df_flat['liasse'].dtype, pl.Struct):
    field_names = [f.name for f in df_flat['liasse'].dtype.fields]
    print(f"DEBUG: Noms des champs dans le Struct: {field_names}")
    
    # Utiliser les vrais noms de champs
    if len(field_names) >= 2:
        code_field = field_names[0]
        value_field = field_names[1]
        print(f"DEBUG: Utilisation des champs '{code_field}' et '{value_field}'")
        
        df_struct = df_flat.select(
            "siren",
            "date_cloture_exercice",
            pl.col("liasse").struct.field(code_field).alias("code_bilan"),
            pl.col("liasse").struct.field(value_field).alias("valeur")
        )
    else:
        print(f"ERREUR: Structure inattendue avec {len(field_names)} champs", file=sys.stderr)
        sys.exit(1)
else:
    print(f"ERREUR: 'liasse' n'est pas un Struct après explode: {df_flat['liasse'].dtype}", file=sys.stderr)
    sys.exit(1)

# --- 6. LE FILTRE "EXPERT" ---
print(f"Filtrage: on ne garde que les {len(CODES_A_GARDER)} codes 'diamant'...")
df_filtered = df_struct.filter(
    pl.col("code_bilan").is_in(CODES_A_GARDER)
)

print(f"DEBUG: {len(df_filtered)} lignes après filtrage")

# --- 7. Le PIVOT (rapide) ---
print("PIVOT 'Expert'...")
# Vérifier s'il y a des doublons avant le pivot
doublons = df_filtered.group_by(["siren", "date_cloture_exercice", "code_bilan"]).agg(
    pl.count().alias("count")
).filter(pl.col("count") > 1)

if len(doublons) > 0:
    print(f"ATTENTION: {len(doublons)} doublons détectés - utilisation de l'agrégation 'sum'")
    print(f"DEBUG: Exemple de doublon: {doublons.head(3)}")

# Utiliser pivot avec agrégation (sum pour additionner les valeurs en cas de doublon)
df_wide = df_filtered.pivot(
    on="code_bilan",
    index=["siren", "date_cloture_exercice"],
    values="valeur",
    aggregate_function="sum"  # Agrégation en cas de doublons
).fill_null(0)

print(f"DEBUG: {len(df_wide)} lignes après pivot")

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

try:
    df_dna_expert.write_parquet(PATH_OUTPUT)
    print(f"DEBUG: {len(df_dna_expert)} lignes sauvegardées")
    print(f"--- Script 02 (Master File INPI) Terminé avec Succès ---")
except Exception as e:
    print(f"ERREUR lors de la sauvegarde: {e}", file=sys.stderr)
    sys.exit(1)