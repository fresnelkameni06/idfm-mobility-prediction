# ============================================================
# ÉTAPE 4 — Modèle 2 : Détection d'anomalies IDFM
# Fichier : src/train_anomaly.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Le notebook 04_anomaly_detection.ipynb sert à explorer.
# Ce fichier est le script de PRODUCTION — il entraîne
# officiellement Isolation Forest et sauvegarde le .pkl.
#
# APPRENTISSAGE NON SUPERVISÉ :
# Pas de variable cible. Le modèle apprend la structure
# des données et repère ce qui s'en écarte.
#
# ENTRÉE  : data/processed/idfm_features.parquet
# SORTIE  : models/anomaly_model.pkl
#           models/anomaly_stats.pkl (moyennes/std par arrêt pour le Z-score)
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
PARQUET_PATH     = "data/processed/idfm_features.parquet"
MODEL_PATH       = "models/anomaly_model.pkl"
STATS_PATH       = "models/anomaly_stats.pkl"

# Hyperparamètres
CONTAMINATION    = 0.02   # 2% d'anomalies attendues (choix métier)
N_ESTIMATORS     = 100    # 100 arbres
SEUIL_Z          = 3      # seuil Z-score

# Features pour Isolation Forest
FEATURES_ISO = ['nb_vald_norm', 'ecart_rolling', 'z_score']


# ============================================================
# FONCTION 1 — Chargement et préparation
# ============================================================
def load_and_prepare(parquet_path: str):
    """
    Charge le Parquet et crée les features pour la détection d'anomalies.

    FEATURES CRÉÉES :
    - z_score       : combien d'écarts-types de la moyenne de l'arrêt
    - nb_vald_norm  : NB_VALD normalisé par arrêt (0 à 1)
    - ecart_rolling : écart à la moyenne mobile 7 jours

    POURQUOI CES 3 FEATURES ?
    Elles décrivent le comportement du trafic sous 3 angles :
    - Volume absolu (normalisé)
    - Écart à la normale statistique (z-score)
    - Écart à la tendance récente (rolling)
    """
    logger.info(f"Chargement : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df['JOUR'] = pd.to_datetime(df['JOUR'])
    df = df.sort_values(['LIBELLE_ARRET', 'JOUR']).reset_index(drop=True)

    # Z-score par arrêt
    moyenne_arret = df.groupby('LIBELLE_ARRET')['NB_VALD'].transform('mean')
    std_arret     = df.groupby('LIBELLE_ARRET')['NB_VALD'].transform('std')
    df['z_score'] = np.where(
        std_arret > 0,
        (df['NB_VALD'] - moyenne_arret) / std_arret,
        0
    )

    # NB_VALD normalisé
    max_arret = df.groupby('LIBELLE_ARRET')['NB_VALD'].transform('max')
    df['nb_vald_norm'] = np.where(max_arret > 0, df['NB_VALD'] / max_arret, 0)

    # Écart à la rolling mean
    df['ecart_rolling'] = (df['NB_VALD'] - df['rolling_mean_7j']).fillna(0)

    logger.info(f"Données préparées : {len(df):,} lignes")
    return df


# ============================================================
# FONCTION 2 — Statistiques par arrêt (pour le Z-score en production)
# ============================================================
def compute_arret_stats(df: pd.DataFrame) -> dict:
    """
    Calcule et sauvegarde moyenne + std + max de chaque arrêt.

    POURQUOI SAUVEGARDER CES STATS ?
    En production, quand l'API reçoit "GARE DU NORD, 800 validations",
    elle doit calculer le Z-score. Pour ça elle a besoin de la moyenne
    et de l'écart-type historique de GARE DU NORD.
    On les pré-calcule ici et on les sauvegarde.
    """
    logger.info("Calcul des statistiques par arrêt...")
    stats = {}
    for arret, grp in df.groupby('LIBELLE_ARRET'):
        stats[arret] = {
            'moyenne': float(grp['NB_VALD'].mean()),
            'std'    : float(grp['NB_VALD'].std()),
            'max'    : float(grp['NB_VALD'].max()),
            'min'    : float(grp['NB_VALD'].min())
        }
    logger.info(f"Statistiques calculées pour {len(stats)} arrêts")
    return stats


# ============================================================
# FONCTION 3 — Entraînement Isolation Forest avec MLflow
# ============================================================
def train_isolation_forest(df: pd.DataFrame):
    """
    Entraîne Isolation Forest et le trace dans MLflow.

    ISOLATION FOREST :
    Algorithme non supervisé qui isole les points anormaux.
    Les anomalies sont isolées rapidement (peu de découpes).
    Les points normaux nécessitent beaucoup de découpes.
    """
    X = df[FEATURES_ISO].fillna(0)

    # Configuration MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
    mlflow.set_experiment("idfm-anomaly-detection")

    logger.info("Entraînement Isolation Forest...")

    with mlflow.start_run(run_name="IsolationForest"):

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

        # Prédictions
        df['pred_iso']     = model.predict(X)          # -1 anomalie, +1 normal
        df['anomalie_iso'] = df['pred_iso'] == -1
        df['score_iso']    = model.score_samples(X)

        # Anomalies Z-score pour comparaison
        df['anomalie_zscore'] = df['z_score'].abs() > SEUIL_Z

        # Statistiques
        nb_anom_iso    = int(df['anomalie_iso'].sum())
        nb_anom_zscore = int(df['anomalie_zscore'].sum())
        nb_consensus   = int((df['anomalie_iso'] & df['anomalie_zscore']).sum())

        # Validation : anomalies sur jours spéciaux
        anom = df[df['anomalie_iso']]
        pct_weekend  = float(anom['is_weekend'].mean() * 100)
        pct_ferie    = float(anom['is_ferie'].mean() * 100)

        # Logging MLflow
        mlflow.log_param("model",          "IsolationForest")
        mlflow.log_param("n_estimators",   N_ESTIMATORS)
        mlflow.log_param("contamination",  CONTAMINATION)
        mlflow.log_param("features",       str(FEATURES_ISO))

        mlflow.log_metric("nb_anomalies_iso",    nb_anom_iso)
        mlflow.log_metric("nb_anomalies_zscore", nb_anom_zscore)
        mlflow.log_metric("nb_consensus",        nb_consensus)
        mlflow.log_metric("pct_anom_weekend",    pct_weekend)
        mlflow.log_metric("pct_anom_ferie",      pct_ferie)

        # Affichage
        print(f"\n{'='*55}")
        print("ISOLATION FOREST — Résultats")
        print(f"{'='*55}")
        print(f"  Anomalies Isolation Forest : {nb_anom_iso:,} "
              f"({nb_anom_iso/len(df)*100:.2f}%)")
        print(f"  Anomalies Z-score          : {nb_anom_zscore:,} "
              f"({nb_anom_zscore/len(df)*100:.2f}%)")
        print(f"  Consensus (les deux)       : {nb_consensus:,}")
        print(f"\n  Validation :")
        print(f"    {pct_weekend:.1f}% des anomalies tombent un week-end")
        print(f"    {pct_ferie:.1f}% des anomalies tombent un jour férié")

    return model, df


# ============================================================
# FONCTION 4 — Sauvegarde
# ============================================================
def save_artifacts(model, stats: dict) -> None:
    """
    Sauvegarde le modèle et les statistiques par arrêt.

    FICHIERS :
    - anomaly_model.pkl : le modèle Isolation Forest
    - anomaly_stats.pkl : moyennes/std par arrêt (pour Z-score en prod)
    """
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    with open(STATS_PATH, 'wb') as f:
        pickle.dump(stats, f)
    logger.info(f"Statistiques sauvegardées : {STATS_PATH}")

    # Vérification
    with open(MODEL_PATH, 'rb') as f:
        model_check = pickle.load(f)
    logger.info(f"Vérification OK — {type(model_check).__name__}")

    print(f"\nFichiers sauvegardés :")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {STATS_PATH}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def train():
    logger.info("=" * 60)
    logger.info("DÉMARRAGE — Entraînement détection d'anomalies")
    logger.info("=" * 60)

    # Étape 1 : chargement et préparation
    df = load_and_prepare(PARQUET_PATH)

    # Étape 2 : statistiques par arrêt
    stats = compute_arret_stats(df)

    # Étape 3 : entraînement Isolation Forest
    model, df = train_isolation_forest(df)

    # Étape 4 : sauvegarde
    save_artifacts(model, stats)

    logger.info("=" * 60)
    logger.info("TERMINÉ — Modèle d'anomalies sauvegardé")
    logger.info("=" * 60)

    print(f"\n{'='*55}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*55}")
    print(f"Modèle       : Isolation Forest")
    print(f"Contamination: {CONTAMINATION} ({CONTAMINATION*100}%)")
    print(f"Sauvegardé   : {MODEL_PATH}")
    print(f"MLflow UI    : mlflow ui")
    print(f"{'='*55}")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    train()