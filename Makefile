# # ===================================================================
# # Makefile
# # ===================================================================

# PYTHON = .venv/bin/python3


# # --- 0. Fichiers de données brutes (dans data/raw/)
# FILE_UL = Data/raw/StockUniteLegale_utf8.parquet
# FILE_ETAB = Data/raw/StockEtablissement_utf8.parquet
# FILE_ETAB_HISTO = Data/raw/StockEtablissementHistorique_utf8.parquet
# FILE_INPI = Data/raw/ExportDetailBilan.parquet

# # URLs stables pour le téléchargement
# URL_UL = https://www.data.gouv.fr/api/1/datasets/r/a29c1297-1f92-4e2a-8f6b-8c902ce96c5f
# URL_ETAB = https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c
# URL_ETAB_HISTO = https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3
# URL_INPI = https://www.data.gouv.fr/api/1/datasets/r/c4ac8f98-2c97-4417-9070-0cbb9de03875

# # Fichiers de données propres (dans data/processed/)
# PROC_SIRENE = Data/processed/sirene_dates.parquet
# PROC_INPI = Data/processed/sirene_bilan.parquet

# # --- 1. SETUP DE L'ENVIRONNEMENT ---

# .venv/bin/activate: requirements.txt
# 	python3 -m venv .venv
# 	. .venv/bin/activate && pip install -r requirements.txt
# 	touch .venv/bin/activate

# setup: .venv/bin/activate
# 	@echo "--- Environnement prêt. ---"

# # --- 2. DATA PROCESSING (La Pipeline "Monstrueuse") ---

# # Fichier 1 : La cohorte SIRENE (dépend des CSV bruts)
# data/processed/cohort_2018_demographic.parquet: scripts/01_build_cohort_sirene.py data/raw/StockUniteLegale_utf8.csv.gz data/raw/StockEtablissement_utf8.csv.gz .venv/bin/activate
# 	@echo "--- [1/2] Lancement du Script 01: Création de la cohorte SIRENE..."
# 	$(PYTHON) scripts/01_build_cohort_sirene.py
# 	@echo "--- [1/2] Cohorte SIRENE créée. ---"

# # Fichier 2 : Le DNA Financier (dépend du Parquet INPI et du Fichier 1)
# data/processed/cohort_2018_FULL_MONSTROUS.parquet: scripts/02_build_dna_financial.py data/raw/donnees-financieres-detaillees-2019.parquet data/processed/cohort_2018_demographic.parquet .venv/bin/activate
# 	@echo "--- [2/2] Lancement du Script 02: Création du 'DNA Financier'..."
# 	$(PYTHON) scripts/02_build_dna_financial.py
# 	@echo "--- [2/2] 'DNA Financier' créé. ---"










# # Cible "process" : Construit les deux fichiers de données
# process: data/processed/cohort_2018_demographic.parquet data/processed/cohort_2018_FULL_MONSTROUS.parquet
# 	@echo "--- Pipeline de données terminée. Fichiers 'processed' prêts. ---"

# # --- 3. COMMANDES PRINCIPALES ---

# # Cible "all" : Installe tout et lance toute la pipeline de données
# all: setup process
# 	@echo "--- TOUT EST PRÊT. Lancez 'jupyter notebook' pour l'analyse. ---"

# # Lancer les notebooks
# notebooks: .venv/bin/activate
# 	@echo "Lancement de Jupyter Notebook..."
# 	. .venv/bin/activate && jupyter notebook

# # Nettoyer le projet (enlever les fichiers générés)
# clean:
# 	@echo "Nettoyage des fichiers générés..."
# 	rm -rf data/processed/*.parquet
# 	rm -rf .ipynb_checkpoints
# 	rm -rf __pycache__
# 	@echo "Nettoyage terminé."

# # Target "phony" pour que 'make' ne confonde pas avec des noms de fichiers
# .PHONY: setup process all notebooks clean

# ===================================================================
# Makefile "Monstrueux" (Auto-Download & Process)
# CORRIGÉ pour respecter la structure "Data" (Majuscule)
# ===================================================================

# --- 1. Variables de Configuration ---
PYTHON = .venv/bin/python3

# Structure de dossier avec 'D' Majuscule
DIR_RAW = Data/raw
DIR_PROC = Data/processed

# Fichiers de données brutes
FILE_UL = $(DIR_RAW)/StockUniteLegale_utf8.parquet
FILE_ETAB = $(DIR_RAW)/StockEtablissement_utf8.parquet
FILE_ETAB_HISTO = $(DIR_RAW)/StockEtablissementHistorique_utf8.parquet
FILE_INPI = $(DIR_RAW)/ExportDetailBilan.parquet

# URLs stables
URL_UL = https://www.data.gouv.fr/api/1/datasets/r/a29c1297-1f92-4e2a-8f6b-8c902ce96c5f
URL_ETAB = https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c
URL_ETAB_HISTO = https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3
URL_INPI = https://www.data.gouv.fr/api/1/datasets/r/c4ac8f98-2c92-4417-9070-0cbb9de03875

# Fichiers de données propres (TES NOMS DE FICHIERS)
PROC_SIRENE = $(DIR_PROC)/sirene_infos.parquet
PROC_INPI = $(DIR_PROC)/sirene_bilan.parquet

# --- 2. Commandes Principales (Mises au début) ---

# Cible "all" : La cible par défaut (lance "make" tout seul)
all: setup process
	@echo "--- TOUT EST PRÊT. Lancez 'make notebooks' pour l'analyse. ---"

# Cible "process" : Construit les deux fichiers de données
process: $(PROC_SIRENE) $(PROC_INPI)
	@echo "--- Pipeline de données terminée. Fichiers 'processed' prêts. ---"

# --- 3. Setup de l'Environnement ---
.venv/bin/activate: requirements.txt
	@echo "--- Création de l'environnement virtuel et installation... ---"
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@touch .venv/bin/activate

setup: .venv/bin/activate
	@echo "--- Environnement prêt. ---"

# --- 4. Création des Dossiers ---
# Cible spéciale pour créer les dossiers "Data" s'ils n'existent pas
$(DIR_RAW) $(DIR_PROC):
	@echo "--- Création des dossiers $(DIR_RAW) et $(DIR_PROC)... ---"
	@mkdir -p $(DIR_RAW)
	@mkdir -p $(DIR_PROC)

# --- 5. Recettes de Téléchargement (La Magie) ---
# 'make' ne lancera ces commandes QUE si les fichiers sont manquants.
# Dépend de la cible de création de dossier.

$(FILE_UL): $(DIR_RAW)
	@echo "Téléchargement de StockUniteLegale..."
	@curl -L "$(URL_UL)" -o "$(FILE_UL)"

$(FILE_ETAB): $(DIR_RAW)
	@echo "Téléchargement de StockEtablissement..."
	@curl -L "$(URL_ETAB)" -o "$(FILE_ETAB)"

$(FILE_ETAB_HISTO): $(DIR_RAW)
	@echo "Téléchargement de StockEtablissementHistorique..."
	@curl -L "$(URL_ETAB_HISTO)" -o "$(FILE_ETAB_HISTO)"

$(FILE_INPI): $(DIR_RAW)
	@echo "Téléchargement de ExportDetailBilan (INPI)..."
	@curl -L "$(URL_INPI)" -o "$(FILE_INPI)"

# --- 6. Data Processing (La Pipeline "Monstrueuse") ---
# Ces cibles dépendent maintenant des fichiers bruts et des scripts

$(PROC_SIRENE): scripts/01_build_sirene_master.py $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) .venv/bin/activate
	@echo "--- [1/2] Lancement Script 01: Création du MASTER FILE SIRENE..."
	$(PYTHON) scripts/01_build_sirene_master.py
	@echo "--- [1/2] Master SIRENE créé. ---"

$(PROC_INPI): scripts/02_build_dna_financial.py $(FILE_INPI) .venv/bin/activate
	@echo "--- [2/2] Lancement Script 02: Création du 'DNA Financier'..."
	$(PYTHON) scripts/02_build_dna_financial.py
	@echo "--- [2/2] 'DNA Financier' créé. ---"

# --- 7. NOUVELLE CIBLE "DOWNLOAD" (Ce que tu veux) ---

download: $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) $(FILE_INPI)
	@echo "--- Téléchargement des 4 fichiers de données brutes terminé. ---"

# --- 8. Autres Commandes ---

# Lancer les notebooks
notebooks: .venv/bin/activate
	@echo "Lancement de Jupyter Notebook..."
	. .venv/bin/activate && jupyter notebook

# Nettoyer le projet (enlever les fichiers générés)
clean:
	@echo "Nettoyage des fichiers générés..."
	rm -rf $(DIR_PROC)/*.parquet
	rm -rf .ipynb_checkpoints
	rm -rf __pycache__
	@echo "Nettoyage terminé."

.PHONY: setup process all notebooks clean download