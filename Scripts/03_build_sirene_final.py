import polars as pl
import pyarrow.parquet as pq
import sys
import os
from datetime import datetime
import re

# --- 1. FONCTION DE LECTURE (POUR LA REUTILISATION) ---
def read_parquet_with_bypass(filepath: str) -> pl.DataFrame:
    """Lit un fichier parquet en utilisant PyArrow puis le convertit en DataFrame Polars."""
    if not os.path.exists(filepath):
        print(f"\n--- ERREUR FATALE: Fichier manquant : {filepath} ---", file=sys.stderr)
        sys.exit(1)
        
    print(f"\n--- Début de la lecture 'bypass' : {filepath} ---")
    
    try:
        table_arrow = pq.read_table(filepath)
        df = pl.from_arrow(table_arrow)
        print(f"Conversion de la table PyArrow en DataFrame Polars... --- SUCCÈS ! ---")
        return df
    except Exception as e:
        print(f"\n--- ERREUR ---", file=sys.stderr)
        print(f"Impossible de lire le fichier, même avec PyArrow : {e}", file=sys.stderr)
        sys.exit(1)

# --- 2. CHEMINS DES FICHIERS ---
# Définit les chemins relatifs à la racine du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

SIRENE_INFOS_PATH = os.path.join(PROJECT_ROOT, "Data/processed/sirene_infos.parquet")
SIRENE_BILAN_PATH = os.path.join(PROJECT_ROOT, "Data/processed/sirene_bilan.parquet")
SIRENE_FINAL_PATH = os.path.join(PROJECT_ROOT, "Data/processed/sirene_final2.parquet")


# =========================================================================
# === TRAITEMENT DU FICHIER DEMOGRAPHIQUE (df_demo) ===
# =========================================================================

df_demo = read_parquet_with_bypass(SIRENE_INFOS_PATH)
print("--- DataFrame df_demo chargé ---")

# Cellule 4 : Filtrage des départements non numériques
df_demo = df_demo.filter(pl.col('departement').str.contains(r'^\d+$'))
print(f"1. Après filtre 'departement': {df_demo.shape}")

# Cellule 6 : Suppression des lignes 'NN' dans trancheEffectifsSiege
df_demo = df_demo.filter(pl.col('trancheEffectifsSiege') != 'NN')
print(f"2. Après filtre 'trancheEffectifsSiege': {df_demo.shape}")

# Cellule 9 : Suppression des lignes 'NN' dans trancheEffectifsUniteLegale
df_demo = df_demo.filter(pl.col('trancheEffectifsUniteLegale') != 'NN')
print(f"3. Après filtre 'trancheEffectifsUniteLegale': {df_demo.shape}")

# Cellule 11 : Suppression des colonnes societeMissionUniteLegale et economieSocialeSolidaireUniteLegale
df_demo = df_demo.drop(['societeMissionUniteLegale', 'economieSocialeSolidaireUniteLegale'])
print(f"4. Après suppression de colonnes: {df_demo.shape}")

# Étape Additionnelle: Création de la colonne anciennete (Code précédent corrigé)
# Ces étapes sont placées ici car elles utilisent la colonne 'anneeCreation' présente dans df_demo
current_year = datetime.now().year
df_demo = df_demo.with_columns(
    (pl.lit(current_year) - pl.col('anneeCreation')).alias('anciennete')
)

# Supprime la colonne anneeCreation (dernière étape de nettoyage de df_demo)
df_demo = df_demo.drop(['anneeCreation', 'moisCreation'])
print(f"5. Après calcul 'anciennete' et suppression 'anneeCreation': {df_demo.shape}")
print("--- df_demo prêt ---")


# =========================================================================
# === TRAITEMENT DU FICHIER FINANCIER (df_bilan) ===
# =========================================================================

df_bilan = read_parquet_with_bypass(SIRENE_BILAN_PATH)
print("--- DataFrame df_bilan chargé ---")

# df_bilan = df_bilan.drop(['date_cloture_exercice'])

# Tri des données : CRUCIAL pour le shift()
df_bilan = df_bilan.sort(["siren", "AnneeClotureExercice"]) 

df_bilan = df_bilan.filter(df_bilan['HN_RésultatNet'] != 0)

siren_counts = df_bilan.filter((df_bilan['AnneeClotureExercice'] >= 2016) & (df_bilan['AnneeClotureExercice'] <= 2022)) \
    .group_by('siren') \
    .agg(pl.count()) \
    .filter(pl.col('count') == 7) \
    .select('siren')
df_bilan = df_bilan.join(siren_counts, on='siren', how='inner')

# Variables de Colonnes à utiliser
COL_RN = 'HN_RésultatNet'

# Calcul des variations N-1 et N-2
df_bilan = df_bilan.with_columns([
    (pl.col(COL_RN) - pl.col(COL_RN).shift(1).over('siren')).alias('variation_resultat_net_N-1'),
    (pl.col(COL_RN) - pl.col(COL_RN).shift(2).over('siren')).alias('variation_resultat_net_N-2'),
])

df_bilan = df_bilan.filter(pl.col('variation_resultat_net_N-2').is_not_null())

df_bilan = df_bilan.with_columns([
    pl.col('HN_RésultatNet').shift(-1).over('siren').alias('Y_RN')
])

df_bilan = df_bilan.filter(pl.col('Y_RN').is_not_null())

df_bilan = df_bilan.drop(['DM_DettesLongTerme', 'DF_CapitauxPropres', 'FB_AchatsMarchandises', 'FA_ChiffreAffairesVentes', 'EG_ImpotsTaxes', 'ratio_marge_brute', 'ratio_capitaux_propres'])

# Cellule 14 : Suppression de la colonne date_cloture_exercice
df_bilan = df_bilan.drop('date_cloture_exercice')
print(f"--- df_bilan prêt --- Shape: {df_bilan.shape}")


# =========================================================================
# === JOINTURE ET SAUVEGARDE FINALE ===
# =========================================================================

print("\n--- Étape Finale: Jointure des deux DataFrames sur 'siren' (Inner Join) ---")

# Jointure pour combiner les données financières et démographiques.
df_final = df_bilan.join(
    df_demo, 
    on="siren", 
    how="inner" # Inner Join pour ne garder que les entreprises présentes dans les deux fichiers
)

print(f"--- Jointure réussie ! --- Shape du DataFrame final: {df_final.shape}")

df_final = df_final.drop(['dateFermeture', 'is_failed_in_3y'])

cols = df_final.columns
cols.remove('siren')
cols.remove('AnneeClotureExercice')
new_order = ['siren', 'AnneeClotureExercice'] + cols
df_final = df_final.select(new_order)

# SAUVEGARDE !
print(f"\n--- Sauvegarde du Master File final dans {SIRENE_FINAL_PATH}...")
os.makedirs(os.path.dirname(SIRENE_FINAL_PATH), exist_ok=True) 

try:
    df_final.write_parquet(SIRENE_FINAL_PATH)
    print(f"DEBUG: {len(df_final)} lignes sauvegardées.")
    print("\n===================================================================")
    print(f"🎉 Script Terminé : {os.path.basename(SIRENE_FINAL_PATH)} créé avec succès.")
    print("===================================================================")
except Exception as e:
    print(f"ERREUR lors de la sauvegarde: {e}", file=sys.stderr)
    sys.exit(1)