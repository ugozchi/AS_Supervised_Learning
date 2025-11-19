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
#     print(f"DEBUG: {len(df_bilan_brut)} lignes lues")
# except Exception as e:
#     print(f"ERREUR: La lecture de {PATH_INPI} a échoué (même avec le bypass).", file=sys.stderr)
#     print(f"Détail: {e}", file=sys.stderr)
#     sys.exit(1)

# # --- DEBUG: Inspecter la structure de la colonne 'liasse' ---
# print("DEBUG: Inspection du schéma de la colonne 'liasse'...")
# print(f"Type de la colonne 'liasse': {df_bilan_brut['liasse'].dtype}")

# # Prendre un échantillon pour voir la structure
# sample = df_bilan_brut.select("liasse").head(1)
# print(f"DEBUG: Premier élément de 'liasse': {sample['liasse'][0]}")

# # Si c'est une structure, afficher les noms des champs
# if isinstance(df_bilan_brut['liasse'].dtype, pl.List):
#     print("DEBUG: 'liasse' est une List")
#     # Exploser pour voir le contenu
#     sample_exploded = df_bilan_brut.head(1).explode("liasse")
#     print(f"DEBUG: Type après explode: {sample_exploded['liasse'].dtype}")
#     print(f"DEBUG: Premier élément après explode: {sample_exploded['liasse'][0]}")
    
#     # Si c'est un Struct, afficher les noms des champs
#     if isinstance(sample_exploded['liasse'].dtype, pl.Struct):
#         field_names = sample_exploded['liasse'].dtype.fields
#         print(f"DEBUG: Noms des champs dans le Struct: {[f.name for f in field_names]}")

# # --- 5. TRANSFORMATION (Explode + Séparation) ---
# print("Transformation (explode) de la liasse...")
# df_flat = df_bilan_brut.explode("liasse")

# # Inspecter la structure après explode
# print(f"DEBUG: Type de 'liasse' après explode: {df_flat['liasse'].dtype}")

# # Vérifier si c'est un Struct et obtenir les noms de champs
# if isinstance(df_flat['liasse'].dtype, pl.Struct):
#     field_names = [f.name for f in df_flat['liasse'].dtype.fields]
#     print(f"DEBUG: Noms des champs dans le Struct: {field_names}")
    
#     # Utiliser les vrais noms de champs
#     if len(field_names) >= 2:
#         code_field = field_names[0]
#         value_field = field_names[1]
#         print(f"DEBUG: Utilisation des champs '{code_field}' et '{value_field}'")
        
#         df_struct = df_flat.select(
#             "siren",
#             "date_cloture_exercice",
#             pl.col("liasse").struct.field(code_field).alias("code_bilan"),
#             pl.col("liasse").struct.field(value_field).alias("valeur")
#         )
#     else:
#         print(f"ERREUR: Structure inattendue avec {len(field_names)} champs", file=sys.stderr)
#         sys.exit(1)
# else:
#     print(f"ERREUR: 'liasse' n'est pas un Struct après explode: {df_flat['liasse'].dtype}", file=sys.stderr)
#     sys.exit(1)

# # --- 6. LE FILTRE "EXPERT" ---
# print(f"Filtrage: on ne garde que les {len(CODES_A_GARDER)} codes 'diamant'...")
# df_filtered = df_struct.filter(
#     pl.col("code_bilan").is_in(CODES_A_GARDER)
# )

# print(f"DEBUG: {len(df_filtered)} lignes après filtrage")

# # --- 7. Le PIVOT (rapide) ---
# print("PIVOT 'Expert'...")
# # Vérifier s'il y a des doublons avant le pivot
# doublons = df_filtered.group_by(["siren", "date_cloture_exercice", "code_bilan"]).agg(
#     pl.count().alias("count")
# ).filter(pl.col("count") > 1)

# if len(doublons) > 0:
#     print(f"ATTENTION: {len(doublons)} doublons détectés - utilisation de l'agrégation 'sum'")
#     print(f"DEBUG: Exemple de doublon: {doublons.head(3)}")

# # Utiliser pivot avec agrégation (sum pour additionner les valeurs en cas de doublon)
# df_wide = df_filtered.pivot(
#     on="code_bilan",
#     index=["siren", "date_cloture_exercice"],
#     values="valeur",
#     aggregate_function="sum"  # Agrégation en cas de doublons
# ).fill_null(0)

# print(f"DEBUG: {len(df_wide)} lignes après pivot")

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

# try:
#     df_dna_expert.write_parquet(PATH_OUTPUT)
#     print(f"DEBUG: {len(df_dna_expert)} lignes sauvegardées")
#     print(f"--- Script 02 (Master File INPI) Terminé avec Succès ---")
# except Exception as e:
#     print(f"ERREUR lors de la sauvegarde: {e}", file=sys.stderr)
#     sys.exit(1)

import polars as pl
import pyarrow.parquet as pq
import sys
import os

# --- 1. GESTION DES CHEMINS & CONFIGURATION ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.join(os.getcwd(), "Scripts") 

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

PATH_INPI = os.path.join(PROJECT_ROOT, "Data/raw/ExportDetailBilan.parquet")
# NOUVEAU CHEMIN DE SORTIE pour l'analyse exploratoire (EDA)
PATH_OUTPUT_EDA = os.path.join(PROJECT_ROOT, "Data/processed/sirene_bilan_EDA.parquet")

# Configuration de la cohorte d'intérêt
ANNEES_REQUISES = {2017, 2018, 2019, 2020}
NOMBRE_ANNEES_REQUIS = len(ANNEES_REQUISES)

print("--- Lancement Script 03: Création du Fichier Bilan (Cohorte EDA 4 ans) ---")

# --- 2. VÉRIFICATION ET DÉFINITIONS ---
if not os.path.exists(PATH_INPI):
    print(f"ERREUR FATALE: Fichier brut manquant : {PATH_INPI}", file=sys.stderr)
    sys.exit(1)

# Codes INPI et mapping de renommage
CODES_A_GARDER = [
    'HN', 'FA', 'FB', 'CJ-CK', 'DL', 'DM', 'DA', 'FJ', 'FR', 'DF', 'EG'
]
RENAMING_MAP = {
    'HN': 'HN_RésultatNet', 'FA': 'FA_ChiffreAffairesVentes', 'FB': 'FB_AchatsMarchandises',
    'CJ-CK': 'CJCK_TotalActifBrut', 'DL': 'DL_DettesCourtTerme', 'DM': 'DM_DettesLongTerme',
    'DA': 'DA_TresorerieActive', 'FJ': 'FJ_ResultatFinancier', 'FR': 'FR_ResultatExceptionnel',
    'DF': 'DF_CapitauxPropres', 'EG': 'EG_ImpotsTaxes'
}
CODE_FIELD_NAME = "code_bilan"
VALUE_FIELD_NAME = "valeur"


# ===================================================================
# PARTIE 1 : CHARGEMENT, PIVOT, ET RATIOS (DNA FINANCIER)
# ===================================================================

def run_transformation_et_ratios():
    """Charge le fichier, explode la liasse, filtre, pivote et calcule les ratios."""
    print("3. Lecture, Explode et Pivot...")
    
    # Lecture (utilisant pyarrow pour gérer les types complexes de la liasse)
    try:
        table_inpi = pq.read_table(PATH_INPI)
        df_bilan_brut = pl.from_arrow(table_inpi)
    except Exception as e:
        print(f"ERREUR lors de la lecture de {PATH_INPI}: {e}", file=sys.stderr)
        sys.exit(1)

    # 3.1. TRANSFORMATION (Explode + Séparation Struct)
    df_flat = df_bilan_brut.explode("liasse")
    
    if isinstance(df_flat['liasse'].dtype, pl.Struct):
        field_names = [f.name for f in df_flat['liasse'].dtype.fields]
        
        df_struct = df_flat.select(
            "siren",
            "date_cloture_exercice",
            pl.col("liasse").struct.field(field_names[0]).alias(CODE_FIELD_NAME),
            pl.col("liasse").struct.field(field_names[1]).alias(VALUE_FIELD_NAME)
        )
    else:
        print(f"ERREUR: 'liasse' n'est pas un Struct après explode. Type: {df_flat['liasse'].dtype}", file=sys.stderr)
        sys.exit(1)

    # 3.2. FILTRE ET PIVOT
    df_filtered = df_struct.filter(pl.col(CODE_FIELD_NAME).is_in(CODES_A_GARDER))

    df_wide = df_filtered.pivot(
        on=CODE_FIELD_NAME,
        index=["siren", "date_cloture_exercice"],
        values=VALUE_FIELD_NAME,
        aggregate_function="sum"
    ).fill_null(0)

    df_wide = df_wide.rename(RENAMING_MAP)

    # 3.3. FEATURE ENGINEERING DES RATIOS
    print("4. Calcul des ratios financiers et ajout de l'année...")
    df_dna_expert = df_wide.with_columns([
        pl.col("date_cloture_exercice").dt.year().alias("AnneeClotureExercice"),
        pl.col("siren").cast(pl.String),
        # Ratios (on garde tout)
        (pl.col("HN_RésultatNet") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_rentabilite_nette"),
        ((pl.col("DL_DettesCourtTerme") + pl.col("DM_DettesLongTerme")) / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_endettement"),
        ((pl.col("FA_ChiffreAffairesVentes") - pl.col("FB_AchatsMarchandises")) / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_marge_brute"),
        (pl.col("DF_CapitauxPropres") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_capitaux_propres"),
        (pl.col("DA_TresorerieActive") / (pl.col("CJCK_TotalActifBrut") + 1e-6)).fill_nan(0).alias("ratio_tresorerie"),
        (pl.col("FJ_ResultatFinancier") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_financier"),
        (pl.col("FR_ResultatExceptionnel") / (pl.col("FA_ChiffreAffairesVentes") + 1e-6)).fill_nan(0).alias("ratio_resultat_exceptionnel")
    ])

    return df_dna_expert.sort(["siren", "date_cloture_exercice"])

# ===================================================================
# PARTIE 2 : FILTRAGE DE COHORTE (4 ans stricts) ET SAUVEGARDE EDA
# ===================================================================

def filter_cohort_and_save(df_bilan):
    """Filtre les SIRENs qui ont exactement 4 bilans pour les années 2017-2020 et sauvegarde le résultat."""
    print(f"5. Filtrage strict sur la cohorte {min(ANNEES_REQUISES)}-{max(ANNEES_REQUISES)}...")

    # 5.1. Filtrer le DataFrame principal pour ne garder que les lignes dans la période d'intérêt (2017-2020)
    df_bilan_periodique = df_bilan.filter(pl.col("AnneeClotureExercice").is_in(ANNEES_REQUISES))

    # 5.2. Identifier les SIRENs qui ont 4 entrées uniques
    sirens_complets = (
        df_bilan_periodique
        .group_by("siren")
        .agg(
            nombre_bilans=pl.len(),
            nombre_annees_uniques=pl.col("AnneeClotureExercice").n_unique()
        )
        .filter(
            # Un SIREN doit avoir 4 lignes et 4 années uniques (pour être strict)
            (pl.col("nombre_bilans") == NOMBRE_ANNEES_REQUIS)
            & (pl.col("nombre_annees_uniques") == NOMBRE_ANNEES_REQUIS)
        )
        .select("siren")
    )

    # 5.3. Joindre les SIRENs complets
    df_filtre = df_bilan_periodique.join(sirens_complets, on="siren", how="inner").sort(["siren", "date_cloture_exercice"])
    
    # 5.4. Sauvegarde
    print(f"   -> Nombre de SIRENs conservés : {sirens_complets.shape[0]}")
    print(f"   -> Nombre total de lignes (4 par SIREN) : {df_filtre.shape[0]}")
    
    # Création du dossier processed
    os.makedirs(os.path.dirname(PATH_OUTPUT_EDA), exist_ok=True)

    try:
        df_filtre.write_parquet(PATH_OUTPUT_EDA)
        print(f"--- Script 03 (Fichier EDA Cohorte) Terminé avec Succès ---")
        print(f"Fichier créé : {PATH_OUTPUT_EDA}")
    except Exception as e:
        print(f"ERREUR lors de la sauvegarde: {e}", file=sys.stderr)
        sys.exit(1)


# ===================================================================
# EXECUTION PRINCIPALE
# ===================================================================

if __name__ == "__main__":
    df_bilan_ratios = run_transformation_et_ratios()
    filter_cohort_and_save(df_bilan_ratios)