# ============================================================
# ÉTAPE 3 — Feature Engineering
# Fichier : src/preprocessing.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Ce fichier contient toutes les fonctions de transformation
# des données brutes en features utilisables par XGBoost.
# On l'appelle "preprocessing" car on pré-traite les données
# AVANT de les envoyer au modèle ML.
#
# Ce fichier est réutilisé par :
# - notebooks/02_feature_engineering.ipynb (pour explorer)
# - src/train_traffic.py (pour entraîner XGBoost)
# - api/main.py (pour prédire en production)
#
# ENTRÉE  : table MySQL 'validations' (468 226 lignes brutes)
# SORTIE  : data/processed/idfm_features.parquet (données enrichies)
# ============================================================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONNEXION MYSQL
# ============================================================
def get_engine():
    load_dotenv()
    DB_USER     = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_HOST     = os.getenv('DB_HOST', 'localhost')
    DB_PORT     = os.getenv('DB_PORT', '3306')
    DB_NAME     = os.getenv('DB_NAME', 'idfm_mobility')
    return create_engine(
        f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )


# ============================================================
# ÉTAPE 3.0 — Chargement depuis MySQL
# ============================================================
def load_from_mysql(engine) -> pd.DataFrame:
    """
    Charge les données brutes depuis MySQL.

    POURQUOI DEPUIS MYSQL ET PAS LE CSV ?
    Parce que MySQL contient les données nettoyées (types corrects,
    BOM supprimé, encodage propre). Le CSV brut avait des problèmes
    qu'on a réglés à l'Étape 1.
    """
    logger.info("Chargement depuis MySQL...")
    df = pd.read_sql('SELECT * FROM validations', con=engine)
    df['JOUR'] = pd.to_datetime(df['JOUR'])
    logger.info(f"Chargé : {len(df):,} lignes")
    return df


# ============================================================
# ÉTAPE 3.1 — Création de is_defini
# ============================================================
def add_is_defini(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la colonne is_defini (True/False).

    POURQUOI :
    Les lignes CATEGORIE_TITRE = 'NON DEFINI' représentent 13.14%
    des données. On les marque avec un booléen pour :
    - Les exclure facilement lors du training XGBoost
    - Les garder pour la détection d'anomalies
    - Les identifier dans le dashboard Power BI

    True  = titre connu (Navigo, Imagine R, etc.)
    False = titre non identifié (NON DEFINI)
    """
    logger.info("Création de is_defini...")
    df['is_defini'] = df['CATEGORIE_TITRE'] != 'NON DEFINI'
    nb_false = (~df['is_defini']).sum()
    logger.info(f"is_defini créé — {nb_false:,} lignes NON DEFINI ({nb_false/len(df)*100:.1f}%)")
    return df


# ============================================================
# ÉTAPE 3.2 — Agrégation par (LIBELLE_ARRET, JOUR)
# ============================================================
def aggregate_by_arret_jour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège NB_VALD par (arrêt, jour) en sommant toutes les catégories.

    POURQUOI AGRÉGER ?
    Notre table a plusieurs lignes par arrêt et par jour
    (une ligne par catégorie de titre). Pour XGBoost, on a besoin
    d'une seule ligne par (arrêt, jour) avec le trafic total.

    Avant agrégation :
      CHATELET | 2025-07-18 | Navigo    | 8000
      CHATELET | 2025-07-18 | Imagine R | 2000
      CHATELET | 2025-07-18 | NON DEFINI| 500

    Après agrégation (sans NON DEFINI) :
      CHATELET | 2025-07-18 | 10000

    POURQUOI EXCLURE LES NON DEFINIS ICI ?
    Pour XGBoost on veut le trafic "réel" — les NON DEFINIS
    faussent le comptage car on ne sait pas si ce sont de vraies
    validations ou des erreurs de lecture.
    """
    logger.info("Agrégation par (LIBELLE_ARRET, JOUR)...")

    # Version SANS NON DEFINIS — pour XGBoost
    df_defini = df[df['is_defini'] == True].copy()

    df_agg = (df_defini
              .groupby(['LIBELLE_ARRET', 'JOUR'])
              .agg(
                  NB_VALD=('NB_VALD', 'sum'),
                  CODE_STIF_TRNS=('CODE_STIF_TRNS', 'first'),
                  CODE_STIF_RES=('CODE_STIF_RES', 'first'),
                  CODE_STIF_ARRET=('CODE_STIF_ARRET', 'first'),
                  ID_ZDC=('ID_ZDC', 'first')
              )
              .reset_index())

    logger.info(f"Agrégation terminée : {len(df_agg):,} lignes (arrêt × jour)")
    return df_agg


# ============================================================
# ÉTAPE 3.3 — Version pivot (une colonne par catégorie de titre)
# ============================================================
def create_pivot_by_titre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une version pivot avec une colonne par catégorie de titre.

    POURQUOI CETTE VERSION PIVOT ?
    Elle permet d'analyser le PROFIL de chaque arrêt selon les titres.
    Par exemple :
    - Un arrêt avec beaucoup d'Imagine R → zone scolaire
    - Un arrêt avec beaucoup de Forfaits courts → zone touristique
    - Un arrêt avec beaucoup d'Amethyste → zone résidentielle senior

    Ces informations enrichissent le clustering (Étape 4 Modèle 3).

    Résultat :
      CHATELET | 2025-07-18 | navigo=8000 | imagine_r=2000 | ...
    """
    logger.info("Création du pivot par catégorie de titre...")

    pivot = (df.groupby(['LIBELLE_ARRET', 'JOUR', 'CATEGORIE_TITRE'])['NB_VALD']
               .sum()
               .unstack(fill_value=0)
               .reset_index())

    # Nettoyer les noms de colonnes
    # (remplacer espaces et caractères spéciaux par _)
    pivot.columns = [
        col.lower()
               .replace(' ', '_')
               .replace("'", '')
               .replace('-', '_')
        if col not in ['LIBELLE_ARRET', 'JOUR'] else col
        for col in pivot.columns
    ]

    logger.info(f"Pivot créé : {len(pivot):,} lignes, {len(pivot.columns)} colonnes")
    return pivot


# ============================================================
# ÉTAPE 3.4 — Variables temporelles
# ============================================================
def add_temporal_features(df: pd.DataFrame,
                           calendrier_path: str = 'data/external/calendrier_feries_fr.csv'
                           ) -> pd.DataFrame:
    """
    Crée toutes les variables temporelles à partir de JOUR.

    POURQUOI CES VARIABLES ?
    Le trafic de transport est TRÈS lié au calendrier.
    Ces variables expliquent ~70% de la variance du trafic.
    XGBoost va les utiliser pour apprendre que :
    - Un dimanche est toujours moins chargé qu'un mardi
    - Août est toujours moins chargé que septembre
    - Un jour férié se comporte comme un dimanche

    VARIABLES CRÉÉES :
    - jour_semaine  : 0=Lundi ... 6=Dimanche (entier pour XGBoost)
    - is_weekend    : True si samedi ou dimanche
    - mois          : 7=juillet, 8=août, 9=septembre
    - semaine_annee : numéro de la semaine dans l'année (1-52)
    - is_ferie      : True si jour férié officiel
    - is_vacances   : True si vacances scolaires
    """
    logger.info("Ajout des variables temporelles...")

    # Extraire depuis JOUR
    # .dt = accesseur datetime de Pandas
    # .dayofweek = 0 (lundi) à 6 (dimanche)
    df['jour_semaine']  = df['JOUR'].dt.dayofweek
    df['is_weekend']    = df['JOUR'].dt.dayofweek >= 5
    df['mois']          = df['JOUR'].dt.month
    df['semaine_annee'] = df['JOUR'].dt.isocalendar().week.astype(int)
    df['annee']         = df['JOUR'].dt.year

    # Charger le calendrier des fériés et vacances
    # POURQUOI UN FICHIER EXTERNE ?
    # Python ne connaît pas le calendrier scolaire français.
    # On lui fournit manuellement les dates dans un CSV.
    try:
        cal = pd.read_csv(calendrier_path)
        cal['date'] = pd.to_datetime(cal['date'])

        jours_feries  = set(cal[cal['type'] == 'ferie']['date'])
        jours_vacances = set(cal[cal['type'].isin(['vacances', 'rentree'])]['date'])

        df['is_ferie']   = df['JOUR'].isin(jours_feries)
        df['is_vacances'] = df['JOUR'].isin(jours_vacances)

        logger.info(f"Calendrier chargé — {len(jours_feries)} jours fériés, "
                    f"{len(jours_vacances)} jours de vacances")
    except FileNotFoundError:
        logger.warning(f"Calendrier non trouvé : {calendrier_path}")
        logger.warning("is_ferie et is_vacances seront False partout")
        df['is_ferie']    = False
        df['is_vacances'] = False

    logger.info("Variables temporelles créées")
    return df


# ============================================================
# ÉTAPE 3.5 — Variables de lag
# ============================================================
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables de lag pour chaque arrêt.

    POURQUOI LES LAGS ?
    Un lag c'est "la valeur passée d'une variable".
    Pour prédire le trafic de demain, le meilleur indicateur
    c'est le trafic d'hier (lag1) et d'il y a 7 jours (lag7).

    POURQUOI LAG7 PLUS QUE LAG1 ?
    Parce que le trafic est cyclique sur 7 jours.
    Un lundi ressemble toujours au lundi de la semaine précédente.
    lag7 capture ce pattern cyclique hebdomadaire.

    VARIABLES CRÉÉES :
    - NB_VALD_lag1      : trafic du même arrêt hier
    - NB_VALD_lag7      : trafic du même arrêt il y a 7 jours
    - rolling_mean_7j   : moyenne des 7 derniers jours (tendance)
    - rolling_std_7j    : écart-type des 7 derniers jours (stabilité)

    POURQUOI ROLLING MEAN ET STD ?
    - rolling_mean_7j = "quelle est la tendance récente ?"
      Si la moyenne monte, le trafic est en hausse
    - rolling_std_7j = "le trafic est-il stable ou erratique ?"
      Un grand écart-type = arrêt très variable (ex: Stade de France)

    IMPORTANT — LES NULL :
    Les 7 premiers jours de chaque arrêt auront des NULL dans lag7
    car il n'y a pas encore 7 jours d'historique.
    On les remplit avec la médiane de l'arrêt (fill_value).
    """
    logger.info("Création des variables de lag...")

    # Trier par arrêt puis par date — OBLIGATOIRE pour les lags
    # POURQUOI : le lag doit prendre la ligne PRÉCÉDENTE dans le temps,
    # pas n'importe quelle ligne
    df = df.sort_values(['LIBELLE_ARRET', 'JOUR']).copy()

    # groupby + shift = lag par arrêt
    # POURQUOI groupby AVANT shift ?
    # Sans groupby, le lag de CHATELET prendrait la dernière ligne
    # de l'arrêt précédent dans l'ordre alphabétique.
    # Avec groupby, le lag reste dans le même arrêt.
    grp = df.groupby('LIBELLE_ARRET')['NB_VALD']

    # shift(1) = décaler d'une ligne vers le bas = valeur d'hier
    df['NB_VALD_lag1'] = grp.shift(1)

    # shift(7) = décaler de 7 lignes = valeur il y a 7 jours
    df['NB_VALD_lag7'] = grp.shift(7)

    # Rolling mean sur 7 jours (fenêtre glissante)
    # min_periods=1 = calculer même si moins de 7 jours disponibles
    df['rolling_mean_7j'] = (grp
                              .transform(lambda x: x.rolling(7, min_periods=1).mean()))

    # Rolling std sur 7 jours
    df['rolling_std_7j'] = (grp
                             .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0)))

    # Gérer les NULL des 10 arrêts incomplets (< 60 jours)
    # POURQUOI : CAMBRONNE n'a que 3 jours → lag7 = NULL sur tout
    # On remplace les NULL par la médiane de l'arrêt
    # C'est mieux que de supprimer ces arrêts
    for col in ['NB_VALD_lag1', 'NB_VALD_lag7']:
        mediane_par_arret = df.groupby('LIBELLE_ARRET')[col].transform('median')
        df[col] = df[col].fillna(mediane_par_arret)

    nb_nulls = df[['NB_VALD_lag1', 'NB_VALD_lag7']].isnull().sum().sum()
    logger.info(f"Lags créés — NULL restants : {nb_nulls}")
    return df


# ============================================================
# ÉTAPE 3.6 — Score de congestion
# ============================================================
def add_congestion_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le score de congestion pour chaque arrêt et chaque jour.

    POURQUOI CE SCORE ?
    Le score = NB_VALD du jour / max historique de cet arrêt
    Il varie entre 0 et 1 :
    - score = 1.0 → l'arrêt est à son maximum absolu (record)
    - score = 0.5 → l'arrêt est à 50% de sa capacité maximale
    - score = 0.1 → l'arrêt est très calme

    UTILITÉ :
    1. Dashboard Power BI : carte colorée par score (rouge = saturé)
    2. Clustering : feature qui distingue les arrêts par niveau d'activité
    3. Détection d'anomalies : un arrêt habituellement à 0.8 qui tombe
       à 0.1 = anomalie évidente

    POURQUOI NE PAS UTILISER NB_VALD DIRECTEMENT ?
    Parce que Saint-Lazare a 200x plus de trafic qu'Ablon.
    Avec NB_VALD brut on ne peut pas comparer les arrêts entre eux.
    Le score normalise et permet la comparaison.
    """
    logger.info("Calcul du score de congestion...")

    # Calculer le maximum historique de chaque arrêt
    # transform('max') = pour chaque ligne, met le max de son groupe
    max_par_arret = df.groupby('LIBELLE_ARRET')['NB_VALD'].transform('max')

    # Score = trafic du jour / maximum historique
    # np.where évite la division par zéro (si max = 0, score = 0)
    df['score_congestion'] = np.where(
        max_par_arret > 0,
        (df['NB_VALD'] / max_par_arret).round(4),
        0
    )

    logger.info(f"Score de congestion calculé — "
                f"min: {df['score_congestion'].min():.3f}, "
                f"max: {df['score_congestion'].max():.3f}")
    return df


# ============================================================
# ÉTAPE 3.7 — Encodage de CODE_STIF_TRNS
# ============================================================
def encode_transporteur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode CODE_STIF_TRNS en variable numérique.

    POURQUOI ENCODER ?
    XGBoost ne peut pas travailler avec des textes ou des codes
    arbitraires. Il faut convertir CODE_STIF_TRNS en quelque chose
    de numérique et ordonné.

    On utilise un label encoding simple :
    chaque code transporteur reçoit un entier unique (0, 1, 2...)
    """
    logger.info("Encodage du transporteur...")
    codes = df['CODE_STIF_TRNS'].astype(str).unique()
    mapping = {code: i for i, code in enumerate(sorted(codes))}
    df['transporteur_encoded'] = df['CODE_STIF_TRNS'].astype(str).map(mapping)
    logger.info(f"Transporteurs encodés : {mapping}")
    return df


# ============================================================
# ÉTAPE 3.8 — Export en Parquet
# ============================================================
def export_to_parquet(df: pd.DataFrame,
                       output_path: str = 'data/processed/idfm_features.parquet') -> None:
    """
    Sauvegarde le DataFrame enrichi en format Parquet.

    POURQUOI PARQUET ET PAS CSV ?
    1. COMPRESSION : Parquet compresse automatiquement les données.
       Un CSV de 100MB peut devenir 15MB en Parquet.
    2. VITESSE : Parquet se lit beaucoup plus vite qu'un CSV.
       Sur 70 000 lignes la différence est notable.
    3. TYPES : Parquet conserve les types de données (int, bool, datetime).
       Un CSV stocke tout comme texte et il faut re-convertir à chaque lecture.
    4. STANDARD : c'est le format standard en data engineering
       (utilisé par Spark, AWS S3, Databricks...).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Parquet sauvegardé : {output_path}")
    logger.info(f"Taille : {size_mb:.1f} MB | Lignes : {len(df):,} | Colonnes : {len(df.columns)}")
    logger.info(f"Colonnes : {list(df.columns)}")


# ============================================================
# PIPELINE COMPLET
# ============================================================
def run_feature_engineering_pipeline() -> pd.DataFrame:
    """
    Lance le pipeline complet de feature engineering.

    ORDRE DES ÉTAPES :
    1. Charger depuis MySQL
    2. Créer is_defini
    3. Agréger par (arrêt, jour)
    4. Ajouter les variables temporelles
    5. Ajouter les lags
    6. Ajouter le score de congestion
    7. Encoder le transporteur
    8. Exporter en Parquet
    """
    logger.info("=" * 60)
    logger.info("DÉMARRAGE — Pipeline Feature Engineering")
    logger.info("=" * 60)

    engine = get_engine()

    # Étape 1 : charger
    df_raw = load_from_mysql(engine)

    # Étape 2 : is_defini
    df_raw = add_is_defini(df_raw)

    # Étape 3 : agréger par (arrêt, jour) — sans NON DEFINIS
    df = aggregate_by_arret_jour(df_raw)

    # Étape 4 : variables temporelles
    df = add_temporal_features(df)

    # Étape 5 : lags
    df = add_lag_features(df)

    # Étape 6 : score de congestion
    df = add_congestion_score(df)

    # Étape 7 : encoder le transporteur
    df = encode_transporteur(df)

    # Étape 8 : export Parquet
    export_to_parquet(df)

    logger.info("=" * 60)
    logger.info("TERMINÉ — Feature Engineering complet")
    logger.info("=" * 60)

    return df


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    df_features = run_feature_engineering_pipeline()
    print("\nAperçu du DataFrame final :")
    print(df_features.head())
    print(f"\nShape : {df_features.shape}")
    print(f"\nColonnes créées : {list(df_features.columns)}")
