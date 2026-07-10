# ============================================================
# ÉTAPE 8 — Monitoring du drift — IDFM Mobility
# Fichier : monitoring/evidently_report.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Il surveille le "data drift" — la dérive des données.
# Avec le temps, les nouvelles données peuvent devenir
# différentes de celles d'entraînement. Le modèle se dégrade.
# Evidently compare RÉFÉRENCE (entraînement) vs ACTUEL (récent)
# et alerte s'il détecte un drift → réentraînement recommandé.
#
# PRINCIPE :
#   Référence (figée) vs Actuel (à jour) → drift ou pas
#
# LES 3 MODÈLES SURVEILLÉS :
#   - Trafic + Anomalies → idfm_features.parquet
#   - Clustering         → arrets_clusters.parquet
#
# LANCER :
#   python monitoring/evidently_report.py
#   Puis ouvrir les rapports HTML générés dans monitoring/
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — chemins des fichiers
# ============================================================
# Référence = données figées de l'entraînement
# Actuel    = données courantes (avec éventuels ajouts)
FEATURES_REF   = "data/processed/idfm_features_reference.parquet"
FEATURES_CUR   = "data/processed/idfm_features.parquet"
CLUSTERS_REF   = "data/processed/arrets_clusters_reference.parquet"
CLUSTERS_CUR   = "data/processed/arrets_clusters.parquet"

OUTPUT_DIR     = "monitoring"

# Seuil : si plus de 30% des colonnes ont dérivé → alerte globale
SEUIL_DRIFT_GLOBAL = 0.3

# Features surveillées pour chaque modèle
FEATURES_TRAFIC = [
    'NB_VALD', 'jour_semaine', 'is_weekend', 'mois',
    'NB_VALD_lag1', 'NB_VALD_lag7', 'rolling_mean_7j',
    'rolling_std_7j', 'score_congestion'
]

FEATURES_ANOMALIE = [
    'NB_VALD', 'rolling_mean_7j', 'rolling_std_7j', 'score_congestion'
]

FEATURES_CLUSTERING = [
    'trafic_moyen_semaine', 'trafic_moyen_weekend',
    'ratio_we_semaine', 'volume_total', 'coef_variation'
]


# ============================================================
# FONCTION — Extraire le verdict de drift d'un rapport
# ============================================================
def extraire_drift(snapshot) -> dict:
    """
    Extrait le nombre et la proportion de colonnes ayant dérivé.

    Evidently calcule, pour chaque feature, si sa distribution
    a changé significativement entre référence et actuel.
    On récupère la proportion de features driftées.
    """
    d = snapshot.dict()
    for m in d.get('metrics', []):
        v = m.get('value')
        # DriftedColumnsCount renvoie un dict {count, share}
        if isinstance(v, dict) and 'share' in v:
            return {
                'nb_colonnes_driftees': int(v['count']),
                'proportion': float(v['share'])
            }
    return {'nb_colonnes_driftees': 0, 'proportion': 0.0}


# ============================================================
# FONCTION — Générer un rapport pour un modèle
# ============================================================
def generer_rapport(nom_modele: str, ref_path: str, cur_path: str,
                    features: list, output_html: str) -> dict:
    """
    Compare référence vs actuel pour un modèle et génère le rapport HTML.

    ENTRÉE :
        nom_modele  : nom lisible (ex: "Trafic")
        ref_path    : chemin du Parquet de référence
        cur_path    : chemin du Parquet actuel
        features    : liste des colonnes à surveiller
        output_html : où sauvegarder le rapport HTML

    SORTIE :
        dict avec le verdict (drift ou pas)
    """
    # Vérifier que les fichiers existent
    if not os.path.exists(ref_path):
        logger.warning(f"[{nom_modele}] Référence introuvable : {ref_path}")
        return {'modele': nom_modele, 'statut': 'REFERENCE_MANQUANTE'}
    if not os.path.exists(cur_path):
        logger.warning(f"[{nom_modele}] Données actuelles introuvables : {cur_path}")
        return {'modele': nom_modele, 'statut': 'DONNEES_MANQUANTES'}

    # Charger les données
    ref = pd.read_parquet(ref_path)
    cur = pd.read_parquet(cur_path)

    # Garder seulement les features surveillées (présentes dans les deux)
    features_ok = [f for f in features if f in ref.columns and f in cur.columns]
    ref = ref[features_ok].dropna()
    cur = cur[features_ok].dropna()

    # Générer le rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    # Sauvegarder le HTML
    snapshot.save_html(output_html)

    # Extraire le verdict
    drift = extraire_drift(snapshot)
    drift_detecte = drift['proportion'] > SEUIL_DRIFT_GLOBAL

    return {
        'modele'      : nom_modele,
        'statut'      : 'DRIFT' if drift_detecte else 'OK',
        'nb_driftees' : drift['nb_colonnes_driftees'],
        'nb_features' : len(features_ok),
        'proportion'  : drift['proportion'],
        'rapport'     : output_html
    }


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def run_monitoring():
    logger.info("=" * 60)
    logger.info("MONITORING DU DRIFT — IDFM Mobility")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resultats = []

    # --- Modèle 1 : Trafic ---
    logger.info("Analyse du modèle TRAFIC...")
    resultats.append(generer_rapport(
        "Trafic", FEATURES_REF, FEATURES_CUR, FEATURES_TRAFIC,
        os.path.join(OUTPUT_DIR, "rapport_trafic.html")
    ))

    # --- Modèle 2 : Anomalies ---
    logger.info("Analyse du modèle ANOMALIES...")
    resultats.append(generer_rapport(
        "Anomalies", FEATURES_REF, FEATURES_CUR, FEATURES_ANOMALIE,
        os.path.join(OUTPUT_DIR, "rapport_anomalies.html")
    ))

    # --- Modèle 3 : Clustering ---
    logger.info("Analyse du modèle CLUSTERING...")
    resultats.append(generer_rapport(
        "Clustering", CLUSTERS_REF, CLUSTERS_CUR, FEATURES_CLUSTERING,
        os.path.join(OUTPUT_DIR, "rapport_clustering.html")
    ))

    # ============================================================
    # AFFICHAGE DU VERDICT — les alertes
    # ============================================================
    print("\n" + "=" * 60)
    print("RÉSULTATS DU MONITORING")
    print("=" * 60)

    drift_global = False
    for r in resultats:
        modele = r['modele']
        statut = r['statut']

        if statut == 'OK':
            print(f"  {modele:<12} : OK — pas de drift "
                  f"({r['nb_driftees']}/{r['nb_features']} features driftées)")
        elif statut == 'DRIFT':
            drift_global = True
            print(f"  {modele:<12} : DRIFT DÉTECTÉ — "
                  f"{r['nb_driftees']}/{r['nb_features']} features driftées "
                  f"({r['proportion']:.0%}) → réentraînement recommandé")
        else:
            print(f"  {modele:<12} : {statut}")

    print("=" * 60)

    if drift_global:
        print("\n>>> ACTION RECOMMANDÉE :")
        print("    Un ou plusieurs modèles ont dérivé.")
        print("    1. Régénérez le Parquet : python src/preprocessing.py")
        print("    2. Réentraînez : python src/train_traffic.py (etc.)")
    else:
        print("\n>>> Aucune action nécessaire — les modèles sont à jour.")

    print("\nRapports HTML générés dans le dossier monitoring/ :")
    for r in resultats:
        if 'rapport' in r:
            print(f"    → {r['rapport']}")

    return resultats


if __name__ == "__main__":     
    run_monitoring()