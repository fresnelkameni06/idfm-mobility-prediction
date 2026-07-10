# ============================================================
# ÉTAPE 1 — Ingestion du CSV IDFM dans MySQL
# Fichier : src/ingestion.py
# ============================================================

# POURQUOI CE FICHIER ?
# MySQL Workbench peut créer la table (create_table.sql),
# mais pour insérer 468 226 lignes, on ne peut pas le faire
# à la main. Ce script Python fait le travail automatiquement :
# il lit le CSV, nettoie les données, puis les insère dans MySQL
# par lots de 10 000 lignes (pour ne pas surcharger la mémoire).

# ============================================================
# IMPORTS — les bibliothèques dont on a besoin
# ============================================================

import pandas as pd          # pour lire et manipuler le CSV
from sqlalchemy import create_engine, text  # pour se connecter à MySQL
import logging               # pour afficher des messages d'avancement
import time                  # pour mesurer la durée d'exécution
from dotenv import load_dotenv  # pour lire les variables d'environnement
import os                    # pour accéder aux variables d'environnement

# ============================================================
# CONFIGURATION DU LOGGING
# ============================================================
# POURQUOI : au lieu de print() partout, le logging est plus
# professionnel — il affiche l'heure, le niveau (INFO, ERROR...)
# et le message. Pratique pour suivre l'avancement sur 468K lignes.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================
# POURQUOI : on ne met JAMAIS les mots de passe en dur dans le code.
# On les stocke dans un fichier .env (jamais poussé sur GitHub)
# et on les lit avec python-dotenv.
# Le fichier .env doit contenir :
#   DB_USER=root
#   DB_PASSWORD=votre_mot_de_passe
#   DB_HOST=localhost
#   DB_PORT=3306
#   DB_NAME=idfm_mobility

load_dotenv()

DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "3306")
DB_NAME     = os.getenv("DB_NAME", "idfm_mobility")

# Chemin vers le fichier CSV brut
CSV_PATH = "data/raw/validations_idfm.csv"

# Nombre de lignes insérées à la fois dans MySQL
# 10 000 = bon compromis entre vitesse et mémoire
BATCH_SIZE = 10_000


# ============================================================
# FONCTION 1 — Connexion à MySQL
# ============================================================
def get_engine():
    """
    Crée et retourne une connexion SQLAlchemy vers MySQL.

    POURQUOI SQLALCHEMY ?
    SQLAlchemy est une bibliothèque qui fait le pont entre Python
    et les bases de données. Elle permet d'insérer un DataFrame
    Pandas directement dans MySQL sans écrire de SQL manuellement.
    """
    # L'URL de connexion suit ce format standard :
    # dialect+driver://user:password@host:port/database
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    engine = create_engine(
        url,
        echo=False,   # echo=True afficherait tout le SQL généré (utile pour débugger)
    )
    return engine


# ============================================================
# FONCTION 2 — Chargement et nettoyage du CSV
# ============================================================
def load_and_clean_csv(path: str) -> pd.DataFrame:
    """
    Charge le CSV IDFM et nettoie les données.

    POURQUOI CE NETTOYAGE ?
    Le CSV a plusieurs problèmes qu'on doit corriger avant insertion :
    1. BOM (\\ufeff) : caractère invisible en début de fichier qui
       corrompt le nom de la première colonne si on ne le gère pas
    2. Séparateur ';' : pas la virgule habituelle, il faut le préciser
    3. JOUR en string : MySQL veut un vrai type DATE, pas du texte
    4. ID_ZDC en float : il y a peut-être des NaN, on les gère
    """
    logger.info(f"Chargement du CSV depuis : {path}")
    start = time.time()

    df = pd.read_csv(
        path,
        sep=';',              # séparateur point-virgule
        encoding='utf-8-sig'  # utf-8-sig gère le BOM automatiquement
                              # c'est équivalent à enlever le \ufeff manuellement
    )

    logger.info(f"CSV chargé : {len(df):,} lignes, {len(df.columns)} colonnes")
    logger.info(f"Colonnes : {list(df.columns)}")

    # --- Nettoyage 1 : conversion de JOUR en type date Python ---
    # POURQUOI : si on laisse JOUR en string, MySQL l'insère comme texte.
    # En le convertissant en date Python, SQLAlchemy comprend
    # que c'est un DATE MySQL et le stocke correctement.
    df['JOUR'] = pd.to_datetime(df['JOUR'], format='%Y-%m-%d').dt.date

    # --- Nettoyage 2 : NB_VALD en entier ---
    # POURQUOI : on s'assure qu'il n'y a pas de décimales ou de NaN
    # NB_VALD doit être un entier propre (c'est un comptage)
    df['NB_VALD'] = pd.to_numeric(df['NB_VALD'], errors='coerce').fillna(0).astype(int)

    # --- Nettoyage 3 : ID_ZDC — gérer les NaN ---
    # POURQUOI : ID_ZDC peut avoir des valeurs manquantes (NaN en float)
    # On les remplace par 0 et on convertit en entier
    df['ID_ZDC'] = pd.to_numeric(df['ID_ZDC'], errors='coerce').fillna(0).astype(int)

    # --- Nettoyage 4 : supprimer les espaces parasites dans les strings ---
    str_cols = ['LIBELLE_ARRET', 'CATEGORIE_TITRE', 'CODE_STIF_RES', 'CODE_STIF_ARRET']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # --- Vérification des valeurs manquantes ---
    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning(f"Valeurs manquantes détectées :\n{nulls[nulls > 0]}")
    else:
        logger.info("Aucune valeur manquante détectée — données propres")

    elapsed = time.time() - start
    logger.info(f"Nettoyage terminé en {elapsed:.1f}s")

    return df


# ============================================================
# FONCTION 3 — Insertion dans MySQL par batch
# ============================================================
def insert_to_mysql(df: pd.DataFrame, engine) -> None:
    """
    Insère le DataFrame dans la table MySQL 'validations' par lots.

    POURQUOI PAR LOTS (BATCH) ?
    Si on insère les 468 226 lignes en une seule fois, on risque :
    - Un timeout de connexion MySQL
    - Une surcharge mémoire
    En découpant en lots de 10 000, chaque insertion est rapide
    et on peut suivre l'avancement en temps réel.

    POURQUOI if_exists='append' ?
    'append' = ajouter les données à la table existante sans la recréer.
    'replace' aurait supprimé et recréé la table (on perdrait les index).
    'fail' aurait planté si la table existe déjà.
    """
    logger.info(f"Début de l'insertion dans MySQL — table 'validations'")
    logger.info(f"Nombre total de lignes à insérer : {len(df):,}")

    start = time.time()
    total = len(df)
    nb_batches = (total // BATCH_SIZE) + 1

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]  # découper le DataFrame en tranches
        batch_num = (i // BATCH_SIZE) + 1

        batch.to_sql(
            name='validations',     # nom de la table MySQL cible
            con=engine,             # connexion SQLAlchemy
            if_exists='append',     # ajouter sans écraser la table
            index=False,            # ne pas insérer l'index Pandas comme colonne
            method='multi'          # insertion multi-lignes = plus rapide
        )

        logger.info(f"Batch {batch_num}/{nb_batches} inséré — {min(i + BATCH_SIZE, total):,}/{total:,} lignes")

    elapsed = time.time() - start
    logger.info(f"Insertion terminée en {elapsed:.1f}s")


# ============================================================
# FONCTION 4 — Vérification finale
# ============================================================
def verify_insertion(engine) -> None:
    """
    Vérifie que toutes les lignes ont bien été insérées.

    POURQUOI VÉRIFIER ?
    Une insertion peut échouer silencieusement sur certains batches.
    On compte les lignes dans MySQL et on compare avec le CSV.
    Le résultat attendu est 468 226 lignes.
    """
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM validations"))
        count = result.fetchone()[0]

    logger.info(f"Vérification MySQL : {count:,} lignes dans la table 'validations'")

    if count == 468_226:
        logger.info("✅ Succès — les 468 226 lignes sont bien présentes dans MySQL")
    else:
        logger.warning(f"⚠️  Attention — attendu 468 226 lignes, trouvé {count:,}")


# ============================================================
# POINT D'ENTRÉE — ce qui s'exécute quand on lance le script
# ============================================================
if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("DÉMARRAGE — Ingestion IDFM → MySQL")
    logger.info("=" * 60)

    # Étape 1 : connexion à MySQL
    engine = get_engine()
    logger.info(f"Connexion MySQL établie : {DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Étape 2 : chargement et nettoyage du CSV
    df = load_and_clean_csv(CSV_PATH)

    # Étape 3 : insertion dans MySQL par batch
    insert_to_mysql(df, engine)

    # Étape 4 : vérification du nombre de lignes
    verify_insertion(engine)

    logger.info("=" * 60)
    logger.info("TERMINÉ — Ingestion complète")
    logger.info("=" * 60)