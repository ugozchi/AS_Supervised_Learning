# # ===================================================================
# # Makefile
# # ===================================================================

# # --- 1. Variables de Configuration ---
# PYTHON = .venv/bin/python3

# # Structure de dossier avec 'D' Majuscule
# DIR_RAW = Data/raw
# DIR_PROC = Data/processed

# # Fichiers de données brutes
# FILE_UL = $(DIR_RAW)/StockUniteLegale_utf8.parquet
# FILE_ETAB = $(DIR_RAW)/StockEtablissement_utf8.parquet
# FILE_ETAB_HISTO = $(DIR_RAW)/StockEtablissementHistorique_utf8.parquet
# FILE_INPI = $(DIR_RAW)/ExportDetailBilan.parquet

# # URLs stables
# URL_UL = https://www.data.gouv.fr/api/1/datasets/r/a29c1297-1f92-4e2a-8f6b-8c902ce96c5f
# URL_ETAB = https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c
# URL_ETAB_HISTO = https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3
# URL_INPI = https://www.data.gouv.fr/api/1/datasets/r/c4ac8f98-2c92-4417-9070-0cbb9de03875

# # Fichiers de données propres (TES NOMS DE FICHIERS)
# PROC_SIRENE = $(DIR_PROC)/sirene_infos.parquet
# PROC_INPI = $(DIR_PROC)/sirene_bilan.parquet

# # --- 2. Commandes Principales (Mises au début) ---

# # Cible "all" : La cible par défaut (lance "make" tout seul)
# all: setup process
# 	@echo "--- TOUT EST PRÊT. Lancez 'make notebooks' pour l'analyse. ---"

# # Cible "process" : Construit les deux fichiers de données
# process: $(PROC_SIRENE) $(PROC_INPI)
# 	@echo "--- Pipeline de données terminée. Fichiers 'processed' prêts. ---"

# # --- 3. Setup de l'Environnement ---
# .venv/bin/activate: requirements.txt
# 	@echo "--- Création de l'environnement virtuel et installation... ---"
# 	python3 -m venv .venv
# 	. .venv/bin/activate && pip install -r requirements.txt
# 	@touch .venv/bin/activate

# setup: .venv/bin/activate
# 	@echo "--- Environnement prêt. ---"

# # --- 4. Création des Dossiers ---
# # Cible spéciale pour créer les dossiers "Data" s'ils n'existent pas
# $(DIR_RAW) $(DIR_PROC):
# 	@echo "--- Création des dossiers $(DIR_RAW) et $(DIR_PROC)... ---"
# 	@mkdir -p $(DIR_RAW)
# 	@mkdir -p $(DIR_PROC)

# # --- 5. Recettes de Téléchargement (La Magie) ---
# # 'make' ne lancera ces commandes QUE si les fichiers sont manquants.
# # Dépend de la cible de création de dossier.

# $(FILE_UL): $(DIR_RAW)
# 	@echo "Téléchargement de StockUniteLegale..."
# 	@curl -L "$(URL_UL)" -o "$(FILE_UL)"

# $(FILE_ETAB): $(DIR_RAW)
# 	@echo "Téléchargement de StockEtablissement..."
# 	@curl -L "$(URL_ETAB)" -o "$(FILE_ETAB)"

# $(FILE_ETAB_HISTO): $(DIR_RAW)
# 	@echo "Téléchargement de StockEtablissementHistorique..."
# 	@curl -L "$(URL_ETAB_HISTO)" -o "$(FILE_ETAB_HISTO)"

# $(FILE_INPI): $(DIR_RAW)
# 	@echo "Téléchargement de ExportDetailBilan (INPI)..."
# 	@curl -L "$(URL_INPI)" -o "$(FILE_INPI)"

# # --- 6. Data Processing (La Pipeline "Monstrueuse") ---
# # Ces cibles dépendent maintenant des fichiers bruts et des scripts

# $(PROC_SIRENE): Scripts/00_build_sirene_infos.py $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) .venv/bin/activate
# 	@echo "--- [1/2] Lancement Script 01: Création du MASTER FILE SIRENE..."
# 	$(PYTHON) Scripts/00_build_sirene_infos.py
# 	@echo "--- [1/2] Master SIRENE créé. ---"

# $(PROC_INPI): Scripts/01_buil_sirene_bilan.py $(FILE_INPI) .venv/bin/activate
# 	@echo "--- [2/2] Lancement Script 02: Création du 'DNA Financier'..."
# 	$(PYTHON) Scripts/01_buil_sirene_bilan.py
# 	@echo "--- [2/2] 'DNA Financier' créé. ---"

# # --- 7. NOUVELLE CIBLE "DOWNLOAD" (Ce que tu veux) ---

# download: $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) $(FILE_INPI)
# 	@echo "--- Téléchargement des 4 fichiers de données brutes terminé. ---"

# # --- 8. Autres Commandes ---

# # Lancer les notebooks
# notebooks: .venv/bin/activate
# 	@echo "Lancement de Jupyter Notebook..."
# 	. .venv/bin/activate && jupyter notebook

# # Nettoyer le projet (enlever les fichiers générés)
# clean:
# 	@echo "Nettoyage des fichiers générés..."
# 	rm -rf $(DIR_PROC)/*.parquet
# 	rm -rf .ipynb_checkpoints
# 	rm -rf __pycache__
# 	@echo "Nettoyage terminé."

# .PHONY: setup process all notebooks clean download
# ===================================================================
# Makefile "Monstrueux" (v4 - URL Corrigées)
# ===================================================================

# --- 1. Variables de Configuration ---
PYTHON = .venv/bin/python3
DIR_RAW = Data/raw
DIR_PROC = Data/processed

# Fichiers bruts (Cibles de Download)
FILE_UL = $(DIR_RAW)/StockUniteLegale_utf8.parquet
FILE_ETAB = $(DIR_RAW)/StockEtablissement_utf8.parquet
FILE_ETAB_HISTO = $(DIR_RAW)/StockEtablissementHistorique_utf8.parquet
FILE_INPI = $(DIR_RAW)/ExportDetailBilan.parquet

# URLs stables
URL_UL = https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3
URL_ETAB = https://www.data.gouv.fr/api/1/datasets/r/a29c1297-1f92-4e2a-8f6b-8c902ce96c5f
URL_ETAB_HISTO = https://www.data.gouv.fr/api/1/datasets/r/2b3a0c79-f97b-46b8-ac02-8be6c1f01a8c
URL_INPI = https://static.data.gouv.fr/resources/donnees-financieres-detaillees-des-entreprises-format-parquet/20250916-061220/export-detail-bilan.parquet

# Scripts (Dépendances)
SCRIPT_SIRENE = Scripts/01_build_sirene_master.py
SCRIPT_INPI = Scripts/02_build_dna_financial.py

# Fichiers propres (Cibles de Process)
PROC_SIRENE = $(DIR_PROC)/sirene_infos.parquet
PROC_INPI = $(DIR_PROC)/sirene_bilan.parquet

# --- 2. Commandes Principales (Mises au début) ---
all: setup process
	@echo "--- TOUT EST PRÊT. Lancez 'make notebooks' pour l'analyse. ---"

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
$(DIR_RAW) $(DIR_PROC):
	@echo "--- Création des dossiers $(DIR_RAW) et $(DIR_PROC)... ---"
	@mkdir -p $(DIR_RAW)
	@mkdir -p $(DIR_PROC)

# --- 5. Recettes de Téléchargement ---
$(FILE_UL): $(DIR_RAW)
	@echo "Téléchargement de StockUniteLegale (le bon)..."
	@curl -L "$(URL_UL)" -o "$(FILE_UL)"
$(FILE_ETAB): $(DIR_RAW)
	@echo "Téléchargement de StockEtablissement (le bon)..."
	@curl -L "$(URL_ETAB)" -o "$(FILE_ETAB)"
$(FILE_ETAB_HISTO): $(DIR_RAW)
	@echo "Téléchargement de StockEtablissementHistorique (le bon)..."
	@curl -L "$(URL_ETAB_HISTO)" -o "$(FILE_ETAB_HISTO)"
$(FILE_INPI): $(DIR_RAW)
	@echo "Téléchargement de ExportDetailBilan (INPI)..."
	@curl -L "$(URL_INPI)" -o "$(FILE_INPI)"

download: $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) $(FILE_INPI)
	@echo "--- Téléchargement des 4 fichiers de données brutes terminé. ---"

# --- 6. Data Processing (La Pipeline "Monstrueuse") ---
$(PROC_SIRENE): $(SCRIPT_SIRENE) $(FILE_UL) $(FILE_ETAB) $(FILE_ETAB_HISTO) .venv/bin/activate
	@echo "\n--- [1/2] Lancement Script 01: Création du MASTER FILE SIRENE..."
	$(PYTHON) $(SCRIPT_SIRENE)
	@echo "--- [1/2] Master SIRENE créé. ---"

$(PROC_INPI): $(SCRIPT_INPI) $(FILE_INPI) .venv/bin/activate
	@echo "\n--- [2/2] Lancement Script 02: Création du 'DNA Financier'..."
	$(PYTHON) $(SCRIPT_INPI)
	@echo "--- [2/2] 'DNA Financier' créé. ---"

# --- 7. Autres Commandes ---
notebooks: .venv/bin/activate
	@echo "Lancement de Jupyter Notebook..."
	. .venv/bin/activate && jupyter notebook

clean:
	@echo "Nettoyage des fichiers générés..."
	rm -rf $(DIR_PROC)/*.parquet
	rm -rf .ipynb_checkpoints
	rm -rf __pycache__
	@echo "Nettoyage terminé."

.PHONY: setup process all notebooks clean download