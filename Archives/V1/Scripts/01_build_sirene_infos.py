import polars as pl
import os
import sys
from datetime import datetime

# --- 1. GESTION DES CHEMINS & CONFIG ---

try:
    # Permet au script de s'exécuter depuis le Makefile ou directement
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Pour un test dans un notebook si nécessaire, mais l'usage principal est en script
    SCRIPT_DIR = os.path.join(os.getcwd(), "Scripts") 

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Chemins des fichiers
PATH_UL = os.path.join(PROJECT_ROOT, "Data/raw/StockUniteLegale_utf8.parquet")
PATH_ETAB = os.path.join(PROJECT_ROOT, "Data/raw/StockEtablissement_utf8.parquet")
PATH_ETAB_HISTO = os.path.join(PROJECT_ROOT, "Data/raw/StockEtablissementHistorique_utf8.parquet")
PATH_OUTPUT_CLEAN = os.path.join(PROJECT_ROOT, "Data/processed/sirene_infos.parquet") 

TODAY = datetime.today().date()

print("--- Lancement Script 02: Création du Jeu de Données ML Final ---")

# --- 2. VÉRIFICATION DES FICHIERS ---
for path in [PATH_UL, PATH_ETAB, PATH_ETAB_HISTO]:
    if not os.path.exists(path):
        print(f"ERREUR FATALE: Fichier brut manquant : {path}", file=sys.stderr)
        print("Assure-toi que les fichiers sont dans 'Data/raw/' (as-tu lancé 'make download'?)", file=sys.stderr)
        sys.exit(1)

# ===================================================================
# ÉTAPES 1 à 3: Définition des DataFrames Lazy
# ===================================================================

## ÉTAPE 1: Unités Légales (Base Features)
print("1. Définition des features de 'StockUniteLegale'...")
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

## ÉTAPE 2: Établissements (Sièges sociaux)
print("2. Définition des features de 'StockEtablissement' (Sièges)...")
df_sieges = pl.scan_parquet(PATH_ETAB).filter(
    pl.col("etablissementSiege") == True
).select(
    "siren", 
    "siret",
    pl.col("codePostalEtablissement").str.slice(0, 2).alias("departement"),
    pl.col("trancheEffectifsEtablissement").alias("trancheEffectifsSiege"),
    pl.col("caractereEmployeurEtablissement").alias("caractereEmployeurSiege")
)

## ÉTAPE 3: Fermetures
print("3. Définition de la date de fermeture ('StockEtablissementHistorique')...")
df_fermetures = pl.scan_parquet(PATH_ETAB_HISTO).filter(
    pl.col("etatAdministratifEtablissement") == 'F'
).select(
    "siret",
    pl.col("dateFin").alias("dateFermeture")
).group_by("siret").agg(
    pl.col("dateFermeture").max() 
)


# ===================================================================
# ÉTAPE 4: Le "Grand Mariage" Lazy (JOIN)
# ===================================================================
print("4. Jointure lazy des 3 tables...")
df_master_lazy = df_base_features.join(df_sieges, on="siren", how="left")
df_master_lazy = df_master_lazy.join(df_fermetures, on="siret", how="left")


# ===================================================================
# ÉTAPE 5: ENRICHISSEMENT (Feature Engineering)
# ===================================================================
print("5. Enrichissement des features (année/mois de création)...")
df_final_lazy = df_master_lazy.select(
    # Les Colonnes à garder pour la suite du traitement
    "siren",
    "siret", # Garder pour la vérification mais peut être droppé plus tard
    "dateCreationUniteLegale",
    "dateFermeture",
    "categorieJuridiqueUniteLegale",
    "trancheEffectifsUniteLegale",
    "activitePrincipaleUniteLegale",
    "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale",
    "societeMissionUniteLegale",
    "departement",
    "trancheEffectifsSiege",
    "caractereEmployeurSiege"
).with_columns([
    # Ajout des features Année/Mois de création
    pl.col("dateCreationUniteLegale").dt.year().alias("anneeCreation"),
    pl.col("dateCreationUniteLegale").dt.month().alias("moisCreation"),
])


# ===================================================================
# ÉTAPE 6: NETTOYAGE DES DONNÉES (Filtrage des incohérences)
# ===================================================================
print("6. Nettoyage des dates incohérentes et des NULL critiques...")

# 6.1. Drop des NULLs sur les features critiques
CRITICAL_COLS_FOR_NULL_DROP = [
    "departement", 
    "dateCreationUniteLegale", 
    "activitePrincipaleUniteLegale", 
    "trancheEffectifsUniteLegale",
    "categorieEntreprise",
    "caractereEmployeurSiege"
]
df_cleaned_lazy = df_final_lazy.drop_nulls(subset=CRITICAL_COLS_FOR_NULL_DROP)

# 6.2. Filtrage des dates de création/fermeture (>= 1970 et cohérentes)
df_cleaned_lazy = df_cleaned_lazy.filter(
    # Date de création doit être >= 1970 et <= Aujourd'hui
    (pl.col("dateCreationUniteLegale").dt.year() >= 1970) & 
    (pl.col("dateCreationUniteLegale") <= pl.lit(TODAY))
).filter(
    # Date de fermeture (si elle existe) doit être > date de création et < Aujourd'hui
    ( pl.col("dateFermeture").is_null() ) | 
    ( (pl.col("dateFermeture") > pl.col("dateCreationUniteLegale")) & 
      (pl.col("dateFermeture") < pl.lit(TODAY)) )
)


# ===================================================================
# ÉTAPE 7: CRÉATION DE LA VARIABLE CIBLE (Target Y)
# ===================================================================
print("7. Création de la Cible (is_failed_in_3y)...")

df_target_lazy = df_cleaned_lazy.with_columns(
    # 7.1. Calcul de la date limite (Création + 3 ans)
    (pl.col("dateCreationUniteLegale").dt.offset_by("3y")).alias("date_limite_3_ans")
).with_columns(
    # 7.2. Calcul de la Cible is_failed_in_3y
    pl.when(
        (pl.col("dateFermeture").is_not_null()) & # Fermé
        (pl.col("dateFermeture") < pl.col("date_limite_3_ans")) # AVANT la date limite de 3 ans
    ).then(1)
    .otherwise(0)
    .alias("is_failed_in_3y")
)

# 7.3. Nettoyage final des colonnes restantes (features catégorielles avec NULL)
# On choisit les colonnes finales et on remplit les NULL restants (sièges, mission, ESS)
COLS_FINAL = [
    "siren", "siret",
    "dateCreationUniteLegale", "dateFermeture", "is_failed_in_3y", # Target + Clés
    "categorieJuridiqueUniteLegale", "trancheEffectifsUniteLegale", 
    "activitePrincipaleUniteLegale", "categorieEntreprise",
    "economieSocialeSolidaireUniteLegale", "societeMissionUniteLegale",
    "anneeCreation", "moisCreation", "departement",
    "trancheEffectifsSiege", "caractereEmployeurSiege"
]

df_ml_ready_lazy = df_target_lazy.select(COLS_FINAL).fill_null("INCONNU")

# ===================================================================
# ÉTAPE 8: COLLECTE ET SAUVEGARDE FINALE
# ===================================================================
print("8. Collecte et sauvegarde du Master File ML final...")

# On s'assure que le dossier 'processed' existe
os.makedirs(os.path.dirname(PATH_OUTPUT_CLEAN), exist_ok=True)

try:
    # On force l'exécution de toute la pipeline lazy ici
    df_collected = df_ml_ready_lazy.collect()
    
    # Suppression des colonnes intermédiaires/non nécessaires au modèle final
    df_collected = df_collected.drop(["siret"])
    
    # Sauvegarde
    df_collected.write_parquet(PATH_OUTPUT_CLEAN)
    
    print(f"\n--- Script 02 (Master File ML) Terminé avec Succès ---")
    print(f"Fichier créé : {PATH_OUTPUT_CLEAN}")
    print(f"Shape finale : {df_collected.shape}")
    print(f"Répartition de la Cible (is_failed_in_3y) :\n{df_collected.get_column('is_failed_in_3y').value_counts()}")
    print("\nAperçu du DataFrame final :")
    print(df_collected.head())

except Exception as e:
    print(f"ERREUR lors de la collecte/sauvegarde: {e}", file=sys.stderr)
    sys.exit(1)