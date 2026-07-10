 ##src/train_clustering.py# ============================================================
# ÉTAPE 4 — Modèle 3 : Clustering des arrêts IDFM
# Fichier : src/train_clustering.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Le notebook 05_clustering.ipynb sert à explorer.
# Ce fichier est le script de PRODUCTION — il entraîne
# officiellement K-Means (2 versions) et sauvegarde les .pkl.
#
# DEUX VERSIONS EN PRODUCTION :
# - K=2 : vue simple (Méga-hubs vs Réseau standard)
# - K=5 : vue métier détaillée (5 profils exploitables)
#
# ENTRÉE  : data/processed/idfm_features.parquet
# SORTIE  : models/clustering_model_k2.pkl
#           models/clustering_model_k5.pkl
#           models/cluster_scaler.pkl
#           data/processed/arrets_clusters.parquet
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
PARQUET_PATH      = "data/processed/idfm_features.parquet"
MODEL_K2_PATH     = "models/clustering_model_k2.pkl"
MODEL_K5_PATH     = "models/clustering_model_k5.pkl"
SCALER_PATH       = "models/cluster_scaler.pkl"
OUTPUT_PARQUET    = "data/processed/arrets_clusters.parquet"

# Features de profil pour le clustering
FEATURES_CLUSTER = ['trafic_moyen_semaine', 'trafic_moyen_weekend',
                    'ratio_we_semaine', 'volume_total', 'coef_variation']

# ============================================================
# NOMS DES PROFILS (déduits de l'analyse des clusters)
# ============================================================
# Ces noms viennent de l'interprétation métier faite dans le notebook.
# Ils correspondent aux caractéristiques moyennes de chaque cluster.

# K=2 : vue simple
PROFILS_K2 = {
    0: "Réseau standard",
    1: "Méga-hubs"
}

# K=5 : vue métier détaillée
# NOTE : les numéros de cluster peuvent varier à chaque entraînement.
# On les réassigne dynamiquement selon les caractéristiques (voir plus bas).
PROFILS_K5_REGLES = {
    'mega_hub'    : "Méga-hubs",
    'grand_pole'  : "Grands pôles multi-usages",
    'standard'    : "Arrêts urbains standards",
    'residentiel' : "Résidentiel domicile-travail",
    'irregulier'  : "Arrêts irréguliers / événementiels"
}


# ============================================================
# FONCTION 1 — Construction des profils par arrêt
# ============================================================
def build_profils(parquet_path: str) -> pd.DataFrame:
    """
    Charge le Parquet journalier et construit un profil par arrêt.

    On passe de 69 271 lignes (arrêt × jour) à 772 lignes (par arrêt)
    en calculant 5 features qui décrivent le comportement de chaque arrêt.
    """
    logger.info(f"Chargement : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df['JOUR'] = pd.to_datetime(df['JOUR'])
    df['is_weekend'] = df['JOUR'].dt.dayofweek >= 5

    logger.info("Construction des profils par arrêt...")
    profils = []
    for arret, grp in df.groupby('LIBELLE_ARRET'):
        semaine = grp[~grp['is_weekend']]['NB_VALD']
        weekend = grp[grp['is_weekend']]['NB_VALD']

        trafic_semaine = semaine.mean() if len(semaine) > 0 else 0
        trafic_weekend = weekend.mean() if len(weekend) > 0 else 0
        ratio_we       = trafic_weekend / trafic_semaine if trafic_semaine > 0 else 0
        volume_total   = grp['NB_VALD'].sum()
        coef_var       = (grp['NB_VALD'].std() / grp['NB_VALD'].mean()
                          if grp['NB_VALD'].mean() > 0 else 0)

        profils.append({
            'LIBELLE_ARRET'       : arret,
            'trafic_moyen_semaine': trafic_semaine,
            'trafic_moyen_weekend': trafic_weekend,
            'ratio_we_semaine'    : ratio_we,
            'volume_total'        : volume_total,
            'coef_variation'      : coef_var
        })

    df_profils = pd.DataFrame(profils)
    logger.info(f"Profils construits : {len(df_profils)} arrêts")
    return df_profils


# ============================================================
# FONCTION 2 — Nommer les clusters K=5 selon leurs caractéristiques
# ============================================================
def nommer_clusters_k5(df_profils: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Attribue un nom métier à chaque cluster selon ses caractéristiques.

    POURQUOI DYNAMIQUE ?
    K-Means ne garantit pas que le cluster 0 soit toujours le même
    d'un entraînement à l'autre. On identifie chaque cluster par
    ses CARACTÉRISTIQUES, pas par son numéro.

    RÈGLES D'IDENTIFICATION :
    - Méga-hubs        : volume_total le plus élevé
    - Grands pôles     : gros volume + ratio_we élevé
    - Résidentiel      : ratio_we le plus faible (calme le week-end)
    - Irréguliers      : coef_variation le plus élevé
    - Standards        : le reste (le plus gros groupe)
    """
    df_temp = df_profils.copy()
    df_temp['cluster'] = labels

    # Caractéristiques moyennes par cluster
    stats = df_temp.groupby('cluster').agg(
        volume=('volume_total', 'mean'),
        ratio=('ratio_we_semaine', 'mean'),
        coef=('coef_variation', 'mean'),
        taille=('LIBELLE_ARRET', 'count')
    )

    noms = {}

    # Méga-hubs = volume max
    c_mega = stats['volume'].idxmax()
    noms[c_mega] = PROFILS_K5_REGLES['mega_hub']

    # Résidentiel = ratio week-end min (parmi les restants)
    restants = stats.drop(c_mega)
    c_resid = restants['ratio'].idxmin()
    noms[c_resid] = PROFILS_K5_REGLES['residentiel']

    # Irréguliers = coef variation max (parmi les restants)
    restants = restants.drop(c_resid)
    c_irreg = restants['coef'].idxmax()
    noms[c_irreg] = PROFILS_K5_REGLES['irregulier']

    # Grands pôles = volume max parmi les 2 restants
    restants = restants.drop(c_irreg)
    c_grand = restants['volume'].idxmax()
    noms[c_grand] = PROFILS_K5_REGLES['grand_pole']

    # Standards = le dernier
    c_std = restants.drop(c_grand).index[0]
    noms[c_std] = PROFILS_K5_REGLES['standard']

    return noms


# ============================================================
# FONCTION 3 — Entraînement des 2 versions avec MLflow
# ============================================================
def train_clustering(df_profils: pd.DataFrame):
    """
    Entraîne K-Means en 2 versions (K=2 et K=5) et trace dans MLflow.
    """
    X = df_profils[FEATURES_CLUSTER].fillna(0)

    # Normalisation (obligatoire pour K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configuration MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
    mlflow.set_experiment("idfm-clustering")

    resultats = {}

    for k, model_path in [(2, MODEL_K2_PATH), (5, MODEL_K5_PATH)]:
        logger.info(f"Entraînement K-Means avec K={k}...")

        with mlflow.start_run(run_name=f"KMeans_k{k}"):

            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)

            # Métriques de qualité
            sil = silhouette_score(X_scaled, labels)
            db  = davies_bouldin_score(X_scaled, labels)
            ch  = calinski_harabasz_score(X_scaled, labels)

            # Logging MLflow
            mlflow.log_param("model", "KMeans")
            mlflow.log_param("n_clusters", k)
            mlflow.log_param("features", str(FEATURES_CLUSTER))
            mlflow.log_metric("silhouette", sil)
            mlflow.log_metric("davies_bouldin", db)
            mlflow.log_metric("calinski_harabasz", ch)

            # Sauvegarde du modèle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            print(f"\n{'='*55}")
            print(f"K-MEANS K={k}")
            print(f"{'='*55}")
            print(f"  Silhouette        : {sil:.4f} (plus haut = mieux)")
            print(f"  Davies-Bouldin    : {db:.4f} (plus bas = mieux)")
            print(f"  Calinski-Harabasz : {ch:.1f} (plus haut = mieux)")

            resultats[k] = {'model': model, 'labels': labels}

            # Répartition des clusters
            print(f"  Répartition :")
            for c in sorted(set(labels)):
                nb = (labels == c).sum()
                print(f"    Cluster {c} : {nb} arrêts")

    return scaler, resultats, X_scaled


# ============================================================
# FONCTION 4 — Sauvegarde du Parquet enrichi + artefacts
# ============================================================
def save_artifacts(df_profils, scaler, resultats):
    """
    Sauvegarde le scaler et le Parquet avec les 2 colonnes de clusters.

    LE PARQUET DE SORTIE contient :
    - Les 5 features de profil de chaque arrêt
    - cluster_k2 : étiquette version 2 clusters + son nom
    - cluster_k5 : étiquette version 5 clusters + son nom
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    # Scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler sauvegardé : {SCALER_PATH}")

    # Ajouter les étiquettes au dataframe
    labels_k2 = resultats[2]['labels']
    labels_k5 = resultats[5]['labels']

    df_profils['cluster_k2'] = labels_k2
    df_profils['cluster_k5'] = labels_k5

    # Nommer les clusters
    # K=2 : le méga-hub = celui avec le plus gros volume
    vol_par_cluster_k2 = df_profils.groupby('cluster_k2')['volume_total'].mean()
    c_mega_k2 = vol_par_cluster_k2.idxmax()
    noms_k2 = {c: (PROFILS_K2[1] if c == c_mega_k2 else PROFILS_K2[0])
               for c in df_profils['cluster_k2'].unique()}

    # K=5 : nommage dynamique selon caractéristiques
    noms_k5 = nommer_clusters_k5(df_profils, labels_k5)

    df_profils['profil_k2'] = df_profils['cluster_k2'].map(noms_k2)
    df_profils['profil_k5'] = df_profils['cluster_k5'].map(noms_k5)

    # Sauvegarde du Parquet
    df_profils.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"Parquet sauvegardé : {OUTPUT_PARQUET}")

    # Afficher les profils K=5
    print(f"\n{'='*55}")
    print("PROFILS K=5 (nommage métier)")
    print(f"{'='*55}")
    for cluster, nom in sorted(noms_k5.items()):
        arrets = df_profils[df_profils['cluster_k5'] == cluster]
        exemples = arrets.nlargest(4, 'volume_total')['LIBELLE_ARRET'].tolist()
        print(f"\n  Cluster {cluster} — {nom} ({len(arrets)} arrêts)")
        print(f"    Exemples : {', '.join(exemples)}")

    return df_profils


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def train():
    logger.info("=" * 60)
    logger.info("DÉMARRAGE — Entraînement clustering des arrêts")
    logger.info("=" * 60)

    # Étape 1 : construire les profils
    df_profils = build_profils(PARQUET_PATH)

    # Étape 2 : entraîner les 2 versions
    scaler, resultats, X_scaled = train_clustering(df_profils)

    # Étape 3 : sauvegarder
    df_final = save_artifacts(df_profils, scaler, resultats)

    logger.info("=" * 60)
    logger.info("TERMINÉ — Modèles de clustering sauvegardés")
    logger.info("=" * 60)

    print(f"\n{'='*55}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*55}")
    print(f"Modèles      : K-Means K=2 et K=5")
    print(f"Sauvegardés  :")
    print(f"  -> {MODEL_K2_PATH}")
    print(f"  -> {MODEL_K5_PATH}")
    print(f"  -> {SCALER_PATH}")
    print(f"  -> {OUTPUT_PARQUET}")
    print(f"MLflow UI    : mlflow ui")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()
