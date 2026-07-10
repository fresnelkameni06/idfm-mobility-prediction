# ============================================================
# ÉTAPE 1 — Ingestion des CSV IDFM dans MySQL
# Fichier : src/ingestion.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Cette version charge les QUATRE trimestres 2025 = une année complète
# (l'ancienne version ne chargeait que le trimestre juil-sept).
#
# POURQUOI UNE ANNÉE COMPLÈTE ?
# Avec 92 jours, le modèle ne voit qu'un été + une rentrée.
# Avec 365 jours, il apprend la vraie saisonnalité annuelle :
# creux d'août, pic de septembre, vacances de Noël, etc.
# C'est indispensable pour le modèle Prophet (prédiction long terme).
#
# LE DÉFI : les 4 fichiers IDFM ne sont PAS homogènes
#   - L'ordre des colonnes change (ID_ZDC avant ou après LIBELLE_ARRET)
#   - Le T4 écrit "Solidarite" SANS accent
#   - Le T2 écrit les transporteurs "100.0" au lieu de "100"
# Ce script harmonise tout avant l'insertion.
#
# LANCER (depuis la racine du projet) :
#   python src/ingestion.py
# ============================================================

import pandas as pd
from sqlalchemy import create_engine, text
import logging
import time
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()

DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "3306")
DB_NAME     = os.getenv("DB_NAME", "idfm_mobility")

# Les 4 fichiers trimestriels, dans l'ordre chronologique
# Tous doivent être dans data/raw/
FICHIERS = [
    ("T1 (jan-mars)",  "data/raw/validations1_idfm.csv"),
    ("T2 (avr-juin)",  "data/raw/validations2_idfm.csv"),
    ("T3 (juil-sept)", "data/raw/validations_idfm.csv"),
    ("T4 (oct-déc)",   "data/raw/validations4_idfm.csv"),
]

BATCH_SIZE = 10_000

# L'ordre des colonnes attendu par la table MySQL
COLONNES_CIBLE = [
    'JOUR', 'CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET',
    'ID_ZDC', 'LIBELLE_ARRET', 'CATEGORIE_TITRE', 'NB_VALD'
]


# ============================================================
# FONCTION 1 — Connexion à MySQL
# ============================================================
def get_engine():
    """Crée une connexion SQLAlchemy vers MySQL."""
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, echo=False)


# ============================================================
# FONCTION 2 — Nettoyer un code transporteur
# ============================================================
def nettoyer_code(valeur) -> str:
    """
    Nettoie un code (transporteur, réseau, arrêt).

    POURQUOI ?
    Le T2 écrit les transporteurs en float : "100.0" au lieu de "100".
    Pandas les lit comme des nombres décimaux. On les remet en entier
    puis en texte, pour que "100.0" et "100" deviennent le même code.
    """
    if pd.isna(valeur):
        return "0"
    # Si c'est un float (100.0), on passe par int pour enlever le .0
    try:
        return str(int(float(valeur)))
    except (ValueError, TypeError):
        return str(valeur).strip()


# ============================================================
# FONCTION 3 — Charger et harmoniser UN fichier
# ============================================================
def load_and_clean_csv(nom: str, path: str) -> pd.DataFrame:
    """
    Charge un CSV trimestriel et l'harmonise au format cible.

    LES 3 HARMONISATIONS :
    1. Colonnes : on sélectionne par NOM (pas par position)
       → gère l'ordre inversé ID_ZDC / LIBELLE_ARRET
    2. Accents : "Solidarite" → "Solidarité" (le T4 n'a pas l'accent)
    3. Codes : "100.0" → "100" (le T2 les écrit en float)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Placez les 4 fichiers trimestriels dans data/raw/"
        )

    logger.info(f"[{nom}] Chargement de {path}")
    start = time.time()

    df = pd.read_csv(
        path,
        sep=';',
        encoding='utf-8-sig',  # gère le BOM \ufeff automatiquement
        dtype=str              # tout en texte d'abord, on typera après
                               # (évite que pandas devine mal les codes)
    )

    # Nettoyer les noms de colonnes (espaces parasites éventuels)
    df.columns = [c.strip() for c in df.columns]

    logger.info(f"[{nom}] {len(df):,} lignes lues — colonnes : {list(df.columns)}")

    # ---- HARMONISATION 1 : sélectionner les colonnes par NOM ----
    # POURQUOI : T1/T2 ont LIBELLE_ARRET avant ID_ZDC,
    #            T3/T4 ont ID_ZDC avant LIBELLE_ARRET.
    # En sélectionnant par nom, l'ordre du fichier n'a plus d'importance.
    manquantes = [c for c in COLONNES_CIBLE if c not in df.columns]
    if manquantes:
        raise ValueError(f"[{nom}] Colonnes manquantes : {manquantes}")

    df = df[COLONNES_CIBLE].copy()

    # ---- Conversion des types ----
    df['JOUR'] = pd.to_datetime(df['JOUR'], format='%Y-%m-%d').dt.date

    df['NB_VALD'] = pd.to_numeric(df['NB_VALD'], errors='coerce').fillna(0).astype(int)
    df['ID_ZDC']  = pd.to_numeric(df['ID_ZDC'],  errors='coerce').fillna(0).astype(int)

    # ---- HARMONISATION 2 : codes transporteur / réseau / arrêt ----
    # "100.0" → "100"
    for col in ['CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET']:
        df[col] = df[col].apply(nettoyer_code)

    # ---- Nettoyage des chaînes ----
    for col in ['LIBELLE_ARRET', 'CATEGORIE_TITRE']:
        df[col] = df[col].astype(str).str.strip()

    # ---- HARMONISATION 3 : l'accent de "Solidarité" ----
    # POURQUOI : le T4 écrit "Contrat Solidarite Transport" (sans accent).
    # Sans correction, on aurait DEUX catégories au lieu d'une,
    # ce qui fausserait toutes les analyses de titres.
    avant = (df['CATEGORIE_TITRE'] == 'Contrat Solidarite Transport').sum()
    df['CATEGORIE_TITRE'] = df['CATEGORIE_TITRE'].replace(
        'Contrat Solidarite Transport',
        'Contrat Solidarité Transport'
    )
    if avant > 0:
        logger.info(f"[{nom}] Accent corrigé sur {avant:,} lignes "
                    f"('Solidarite' → 'Solidarité')")

    # ---- Contrôle qualité ----
    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning(f"[{nom}] Valeurs manquantes :\n{nulls[nulls > 0]}")

    jours = df['JOUR'].nunique()
    arrets = df['LIBELLE_ARRET'].nunique()
    logger.info(f"[{nom}] OK — {len(df):,} lignes | {jours} jours | "
                f"{arrets} arrêts | {time.time() - start:.1f}s")

    return df


# ============================================================
# FONCTION 4 — Charger les 4 trimestres et les concaténer
# ============================================================
def charger_annee_complete() -> pd.DataFrame:
    """
    Charge les 4 fichiers trimestriels et les assemble en un seul DataFrame.

    RÉSULTAT ATTENDU : ~1 892 000 lignes couvrant 365 jours (année 2025).
    """
    dfs = []
    for nom, path in FICHIERS:
        dfs.append(load_and_clean_csv(nom, path))

    logger.info("Concaténation des 4 trimestres...")
    df = pd.concat(dfs, ignore_index=True)

    # Trier par date puis par arrêt (utile pour les lags plus tard)
    df = df.sort_values(['JOUR', 'LIBELLE_ARRET']).reset_index(drop=True)

    # ---- Détection de doublons ----
    # POURQUOI : si un trimestre chevauche l'autre, on aurait des doublons.
    # Une ligne unique = (JOUR, LIBELLE_ARRET, CATEGORIE_TITRE)
    cles = ['JOUR', 'LIBELLE_ARRET', 'CATEGORIE_TITRE']
    doublons = df.duplicated(subset=cles).sum()
    if doublons > 0:
        logger.warning(f"{doublons:,} doublons détectés — suppression")
        df = df.drop_duplicates(subset=cles, keep='first').reset_index(drop=True)

    return df


# ============================================================
# FONCTION 5 — Vider la table avant rechargement
# ============================================================
def vider_table(engine) -> None:
    """
    Vide la table validations avant de recharger l'année complète.

    POURQUOI VIDER ?
    La table contient déjà le T3 (468 226 lignes). Si on ajoutait
    les 4 trimestres par-dessus, le T3 serait en double.
    On repart propre : TRUNCATE efface tout mais garde la structure
    et les index (contrairement à DROP TABLE).
    """
    logger.info("Vidage de la table 'validations' (TRUNCATE)...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE validations"))
        conn.commit()
    logger.info("Table vidée — prête pour le rechargement")


# ============================================================
# FONCTION 6 — Insertion par batch
# ============================================================
def insert_to_mysql(df: pd.DataFrame, engine) -> None:
    """
    Insère le DataFrame dans MySQL par lots de 10 000 lignes.

    Sur ~1.9M lignes, l'insertion prend plusieurs minutes.
    Les logs affichent l'avancement tous les 10 batches.
    """
    logger.info(f"Insertion de {len(df):,} lignes dans 'validations'")
    start = time.time()
    total = len(df)
    nb_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        batch.to_sql(
            name='validations',
            con=engine,
            if_exists='append',
            index=False,
            method='multi'
        )

        # Log tous les 10 batches pour ne pas noyer le terminal
        if batch_num % 10 == 0 or batch_num == nb_batches:
            pct = 100 * min(i + BATCH_SIZE, total) / total
            logger.info(f"Batch {batch_num}/{nb_batches} — "
                        f"{min(i + BATCH_SIZE, total):,}/{total:,} ({pct:.0f}%)")

    logger.info(f"Insertion terminée en {time.time() - start:.1f}s")


# ============================================================
# FONCTION 7 — Vérification finale
# ============================================================
def verify_insertion(engine, attendu: int) -> None:
    """
    Vérifie le contenu de MySQL après insertion.

    On contrôle : nombre de lignes, nombre de jours, plage de dates,
    nombre d'arrêts et de catégories (l'accent doit être harmonisé).
    """
    with engine.connect() as conn:
        count   = conn.execute(text("SELECT COUNT(*) FROM validations")).fetchone()[0]
        jours   = conn.execute(text("SELECT COUNT(DISTINCT JOUR) FROM validations")).fetchone()[0]
        arrets  = conn.execute(text("SELECT COUNT(DISTINCT LIBELLE_ARRET) FROM validations")).fetchone()[0]
        cats    = conn.execute(text("SELECT COUNT(DISTINCT CATEGORIE_TITRE) FROM validations")).fetchone()[0]
        dmin    = conn.execute(text("SELECT MIN(JOUR) FROM validations")).fetchone()[0]
        dmax    = conn.execute(text("SELECT MAX(JOUR) FROM validations")).fetchone()[0] 

    print("\n" + "=" * 60)
    print("VÉRIFICATION MYSQL")
    print("=" * 60)
    print(f"  Lignes            : {count:,}")
    print(f"  Jours distincts   : {jours}")
    print(f"  Plage de dates    : {dmin} → {dmax}")
    print(f"  Arrêts distincts  : {arrets}")
    print(f"  Catégories titres : {cats}  (doit être 7 — sinon accent non harmonisé)")
    print("=" * 60)

    if count == attendu:
        logger.info(f"Succès — {count:,} lignes insérées comme attendu")
    else:
        logger.warning(f"Attendu {attendu:,} lignes, trouvé {count:,}")

    if cats != 7:
        logger.warning(f"{cats} catégories au lieu de 7 — "
                       f"vérifier l'harmonisation des accents")
    if jours != 365:
        logger.warning(f"{jours} jours au lieu de 365 — "
                       f"vérifier que les 4 trimestres sont bien chargés")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("INGESTION ANNUELLE 2025 — 4 trimestres → MySQL")
    logger.info("=" * 60)

    engine = get_engine()
    logger.info(f"Connexion MySQL : {DB_HOST}:{DB_PORT}/{DB_NAME}")

    # 1. Charger et harmoniser les 4 trimestres
    df = charger_annee_complete()
    logger.info(f"Année complète assemblée : {len(df):,} lignes | "
                f"{df['JOUR'].nunique()} jours | "
                f"{df['LIBELLE_ARRET'].nunique()} arrêts")

    # 2. Vider la table (le T3 seul y est encore)
    vider_table(engine)

    # 3. Insérer l'année complète
    insert_to_mysql(df, engine)

    # 4. Vérifier
    verify_insertion(engine, attendu=len(df))

    logger.info("=" * 60)
    logger.info("TERMINÉ — Prochaine étape : python src/preprocessing.py")
    logger.info("=" * 60)