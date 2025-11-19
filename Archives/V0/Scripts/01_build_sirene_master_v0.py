import polars as pl
import sys
import os

# --- 1. GESTION DES CHEMINS ---
# Permet au script de s'exécuter depuis le Makefile
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.join(os.getcwd(), "Scripts") # Si on teste dans un notebook

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


PATH_UL = os.path.join(PROJECT_ROOT, "Data/raw/StockUniteLegale_utf8.parquet")
PATH_ETAB = os.path.join(PROJECT_ROOT, "Data/raw/StockEtablissement_utf8.parquet")
PATH_ETAB_HISTO = os.path.join(PROJECT_ROOT, "Data/raw/StockEtablissementHistorique_utf8.parquet")
PATH_OUTPUT = os.path.join(PROJECT_ROOT, "Data/processed/sirene_infos.parquet") # Cible du Makefile

print("--- Lancement Script 01: Création du MASTER FILE SIRENE ---")

# --- 2. VÉRIFICATION DES FICHIERS ---
for path in [PATH_UL, PATH_ETAB, PATH_ETAB_HISTO]:
    if not os.path.exists(path):
        print(f"ERREUR FATALE: Fichier brut manquant : {path}", file=sys.stderr)
        print("Assure-toi que les fichiers sont dans 'Data/raw/' (as-tu lancé 'make download'?)", file=sys.stderr)
        sys.exit(1)

# ===================================================================
# ÉTAPE 1: La Base (FEATURES X) - Fichier 'StockUniteLegale'
# ===================================================================
print("Étape 1: Lecture des features de 'StockUniteLegale'...")
df_base_features = pl.scan_parquet(PATH_UL).select(
    "siren",
    "dateCreationUniteLegale",
    "categorieJuridiqueUniteLegale",
    "trancheEffectifsUniteLegale",
    "activitePrincipaleUniteLegale",
    "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale",
    "societeMissionUniteLegale"
)

# ===================================================================
# ÉTAPE 2: Trouver le SIRET du Siège (HQ) - Fichier 'StockEtablissement'
# ===================================================================
print("Étape 2: Lecture de 'StockEtablissement' pour trouver les sièges...")
print(f"DEBUG: Chemin utilisé = {PATH_ETAB}")

# Vérification des colonnes disponibles (debug)
try:
    test_cols = pl.scan_parquet(PATH_ETAB).collect_schema()
    print(f"DEBUG: Nombre de colonnes dans le fichier = {len(test_cols.names())}")
    
    # Vérifiez si etablissementSiege existe
    if "etablissementSiege" not in test_cols.names():
        print("ERREUR: La colonne 'etablissementSiege' n'existe pas dans ce fichier!")
        print("Colonnes disponibles:", test_cols.names()[:20])  # Affiche les 20 premières
        sys.exit(1)
    else:
        print("DEBUG: Colonne 'etablissementSiege' trouvée ✓")
        
except Exception as e:
    print(f"ERREUR lors de la vérification du schéma: {e}", file=sys.stderr)
    sys.exit(1)

# Lecture des sièges
df_sieges = pl.scan_parquet(PATH_ETAB).filter(
    pl.col("etablissementSiege") == True
).select(
    "siren", 
    "siret",
    pl.col("codePostalEtablissement").str.slice(0, 2).alias("departement")
)

# ===================================================================
# ÉTAPE 3: Trouver la Date de Fermeture (La Cible Y) - Fichier 'StockEtablissementHistorique'
# ===================================================================
print("Étape 3: Lecture de 'StockEtablissementHistorique' pour trouver les 'morts'...")
print(f"DEBUG: Chemin utilisé = {PATH_ETAB_HISTO}")

df_fermetures = pl.scan_parquet(PATH_ETAB_HISTO).filter(
    pl.col("etatAdministratifEtablissement") == 'F'
).select(
    "siret",
    pl.col("dateFin").alias("dateFermeture")
).group_by("siret").agg(
    pl.col("dateFermeture").max() # On prend la date de fermeture la plus récente
)

# ===================================================================
# ÉTAPE 4: Le "Grand Mariage" SIRENE
# ===================================================================
print("Étape 4: Jointure finale des 3 tables...")
df_master = df_base_features.join(df_sieges, on="siren", how="left")
df_master = df_master.join(df_fermetures, on="siret", how="left")

# ===================================================================
# ÉTAPE 5: Sauvegarde
# ===================================================================
print(f"Sauvegarde du Master File SIRENE dans {PATH_OUTPUT}...")
df_final = df_master.select(
    "siren",
    "dateCreationUniteLegale",
    "dateFermeture",
    "categorieJuridiqueUniteLegale",
    "trancheEffectifsUniteLegale",
    "activitePrincipaleUniteLegale",
    "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale",
    "societeMissionUniteLegale",
    "departement"
)

# On s'assure que le dossier 'processed' existe
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)

# On collecte et on sauvegarde
print("DEBUG: Début de la collecte des données...")
try:
    df_collected = df_final.collect()
    print(f"DEBUG: {len(df_collected)} lignes collectées")
    df_collected.write_parquet(PATH_OUTPUT)
    print(f"--- Script 01 (Master File SIRENE) Terminé avec Succès ---")
except Exception as e:
    print(f"ERREUR lors de la collecte/sauvegarde: {e}", file=sys.stderr)
    sys.exit(1)