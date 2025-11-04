# ===================================================================
# Makefile pour le Projet SIRENE Survival
# Télécommande pour l'installation, le data processing et le modeling.
# ===================================================================

# Nom de l'interpréteur python dans l'env virtuel
# On le force à utiliser python3.11 ou ce que tu veux
PYTHON = .venv/bin/python3

# --- 1. SETUP DE L'ENVIRONNEMENT ---

.venv/bin/activate: requirements.txt
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	touch .venv/bin/activate

setup: .venv/bin/activate
	@echo "--- Environnement prêt. ---"

# --- 2. DATA PROCESSING (La Pipeline "Monstrueuse") ---

# Fichier 1 : La cohorte SIRENE (dépend des CSV bruts)
data/processed/cohort_2018_demographic.parquet: scripts/01_build_cohort_sirene.py data/raw/StockUniteLegale_utf8.csv.gz data/raw/StockEtablissement_utf8.csv.gz .venv/bin/activate
	@echo "--- [1/2] Lancement du Script 01: Création de la cohorte SIRENE..."
	$(PYTHON) scripts/01_build_cohort_sirene.py
	@echo "--- [1/2] Cohorte SIRENE créée. ---"

# Fichier 2 : Le DNA Financier (dépend du Parquet INPI et du Fichier 1)
data/processed/cohort_2018_FULL_MONSTROUS.parquet: scripts/02_build_dna_financial.py data/raw/donnees-financieres-detaillees-2019.parquet data/processed/cohort_2018_demographic.parquet .venv/bin/activate
	@echo "--- [2/2] Lancement du Script 02: Création du 'DNA Financier'..."
	$(PYTHON) scripts/02_build_dna_financial.py
	@echo "--- [2/2] 'DNA Financier' créé. ---"

# Cible "process" : Construit les deux fichiers de données
process: data/processed/cohort_2018_demographic.parquet data/processed/cohort_2018_FULL_MONSTROUS.parquet
	@echo "--- Pipeline de données terminée. Fichiers 'processed' prêts. ---"

# --- 3. COMMANDES PRINCIPALES ---

# Cible "all" : Installe tout et lance toute la pipeline de données
all: setup process
	@echo "--- TOUT EST PRÊT. Lancez 'jupyter notebook' pour l'analyse. ---"

# Lancer les notebooks
notebooks: .venv/bin/activate
	@echo "Lancement de Jupyter Notebook..."
	. .venv/bin/activate && jupyter notebook

# Nettoyer le projet (enlever les fichiers générés)
clean:
	@echo "Nettoyage des fichiers générés..."
	rm -rf data/processed/*.parquet
	rm -rf .ipynb_checkpoints
	rm -rf __pycache__
	@echo "Nettoyage terminé."

# Target "phony" pour que 'make' ne confonde pas avec des noms de fichiers
.PHONY: setup process all notebooks clean