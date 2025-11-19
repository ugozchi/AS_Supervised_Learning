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