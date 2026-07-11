# ============================================================
# Fichier : src/predict.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# C'est le pont entre les modèles entraînés (.pkl)
# et l'API FastAPI (Étape 6).
#
# Il contient 3 fonctions de prédiction — une par modèle :
#   - predict_traffic()  → Modèle 1 : prédiction du trafic
#   - predict_anomaly()  → Modèle 2 : détection d'anomalies
#   - predict_cluster()  → Modèle 3 : clustering des arrêts
#
# L'API FastAPI importe ce fichier et appelle la bonne
# fonction selon l'endpoint demandé :
#   POST /predict/traffic  → predict_traffic()
#   POST /detect/anomaly   → predict_anomaly()
#   GET  /clusters/{arret} → predict_cluster()
#
# PRINCIPE : ce fichier ne sait pas comment les modèles
# ont été entraînés — il charge juste les .pkl et prédit.
# C'est la séparation des responsabilités. 
# ============================================================

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CHEMINS DES MODÈLES
# ============================================================
# Ces chemins sont relatifs à la racine du projet
# L'API FastAPI doit être lancée depuis la racine
MODEL_TRAFFIC_PATH   = "models/traffic_model.pkl"
MODEL_ANOMALY_PATH   = "models/anomaly_model.pkl"
MODEL_CLUSTER_K2_PATH = "models/clustering_model_k2.pkl"
MODEL_CLUSTER_K5_PATH = "models/clustering_model_k5.pkl"
CLUSTER_SCALER_PATH   = "models/cluster_scaler.pkl"
CLUSTERS_PARQUET_PATH = "data/processed/arrets_clusters.parquet"
FEATURE_NAMES_PATH   = "models/feature_names.pkl"
LABEL_ENCODER_PATH   = "models/label_encoder.pkl"
PARQUET_PATH         = "data/processed/idfm_features.parquet"
EXPLAINER_PATH = "models/shap_explainer.pkl"
PROPHET_FERIES_PATH = "models/prophet_feries.pkl"
CALENDRIER_PATH = "data/external/calendrier_feries_fr.csv"


# ============================================================
# CHARGEMENT DES ARTEFACTS
# ============================================================
# On charge les modèles UNE SEULE FOIS au démarrage
# POURQUOI : charger un .pkl à chaque prédiction serait
# très lent (surtout RandomForest qui est lourd)
# En les chargeant une fois, les prédictions sont instantanées

def _load_artifact(path: str):
    """Charge un fichier .pkl depuis le disque."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Avez-vous lancé le script d'entraînement correspondant ?"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


# Chargement au démarrage du module
# Ces variables sont partagées par toutes les fonctions
try:
    _traffic_model   = _load_artifact(MODEL_TRAFFIC_PATH)
    _feature_names   = _load_artifact(FEATURE_NAMES_PATH)
    _label_encoder   = _load_artifact(LABEL_ENCODER_PATH)
    logger.info("Modèle trafic chargé OK")
except FileNotFoundError as e:
    logger.warning(f"Modèle trafic non disponible : {e}")
    _traffic_model = _feature_names = _label_encoder = None

# Chargement de l'explainer SHAP (explicabilité des prédictions)
try:
    _shap_explainer = _load_artifact(EXPLAINER_PATH)
    logger.info("Explainer SHAP chargé OK")
except FileNotFoundError:
    logger.warning("Explainer SHAP non disponible — lancez train_traffic.py")
    _shap_explainer = None

try:
    _anomaly_model = _load_artifact(MODEL_ANOMALY_PATH)
    logger.info("Modèle anomalie chargé OK")
except FileNotFoundError:
    logger.warning("Modèle anomalie non disponible — lancez train_anomaly.py")
    _anomaly_model = None

try:
    _cluster_model_k2 = _load_artifact(MODEL_CLUSTER_K2_PATH)
    _cluster_model_k5 = _load_artifact(MODEL_CLUSTER_K5_PATH)
    _cluster_scaler   = _load_artifact(CLUSTER_SCALER_PATH)
    logger.info("Modèles clustering (k2 + k5) chargés OK")
except FileNotFoundError:
    logger.warning("Modèles clustering non disponibles — lancez train_clustering.py")
    _cluster_model_k2 = _cluster_model_k5 = _cluster_scaler = None

# Calendrier des fériés au format Prophet (pour la prédiction long terme)
# POURQUOI LE CHARGER ICI ?
# predict_long_term() entraîne Prophet à la volée. Reconstruire le
# calendrier à chaque requête serait du gaspillage — on le charge une fois.
try:
    _prophet_feries = _load_artifact(PROPHET_FERIES_PATH)
    logger.info("Calendrier Prophet chargé OK")
except FileNotFoundError:
    logger.warning("Calendrier Prophet non disponible — lancez train_prophet.py")
    _prophet_feries = None

# Parquet des clusters : lecture rapide du cluster des arrêts connus
try:
    _df_clusters = pd.read_parquet(CLUSTERS_PARQUET_PATH)
    logger.info(f"Parquet clusters chargé : {len(_df_clusters)} arrêts")
except FileNotFoundError:
    logger.warning("Parquet clusters non disponible — lancez train_clustering.py")
    _df_clusters = None


# ============================================================
# DONNÉES HISTORIQUES (pour construire les features de lag)
# ============================================================
# POURQUOI CHARGER LE PARQUET ICI ?
# Pour prédire le trafic de demain à CHATELET, le modèle
# a besoin de lag1 (trafic d'hier) et lag7 (trafic il y a 7 jours).
# Ces valeurs viennent des données historiques du Parquet.
# Sans ça, on ne peut pas construire les features pour XGBoost.

try:
    _df_historique = pd.read_parquet(PARQUET_PATH)
    _df_historique['JOUR'] = pd.to_datetime(_df_historique['JOUR'])
    logger.info(f"Données historiques chargées : {len(_df_historique):,} lignes")
except FileNotFoundError:
    logger.warning("Parquet non disponible")
    _df_historique = None


# ============================================================
# CALENDRIER FÉRIÉS / VACANCES (pour _build_features)
# ============================================================
# POURQUOI CHARGER LE CALENDRIER ICI ?
# _build_features doit remplir is_ferie et is_vacances EXACTEMENT
# comme preprocessing.py l'a fait à l'entraînement. Si les deux
# divergent, le modèle est entraîné sur une définition et interrogé
# sur une autre — ses prédictions seraient faussées les jours fériés
# et pendant les vacances. On lit donc le même fichier.
try:
    _cal = pd.read_csv(CALENDRIER_PATH)
    _cal['date'] = pd.to_datetime(_cal['date'])
    _JOURS_FERIES   = set(_cal.loc[_cal['type'] == 'ferie', 'date'])
    _JOURS_VACANCES = set(_cal.loc[_cal['type'].isin(['vacances', 'rentree']), 'date'])
    logger.info(f"Calendrier chargé : {len(_JOURS_FERIES)} fériés, "
                f"{len(_JOURS_VACANCES)} jours de vacances")
except FileNotFoundError:
    logger.warning(f"Calendrier non trouvé : {CALENDRIER_PATH} — "
                   f"is_ferie et is_vacances seront toujours 0")
    _JOURS_FERIES = set()
    _JOURS_VACANCES = set()


# ============================================================
# FONCTION UTILITAIRE — Construire les features pour une prédiction
# ============================================================
def _build_features(arret: str, date_pred: date) -> pd.DataFrame:
    """
    Construit le vecteur de features pour une prédiction.

    POURQUOI CETTE FONCTION ?
    Le modèle a besoin de 12 features pour chaque prédiction.
    Cette fonction les calcule automatiquement depuis :
    - La date demandée (pour les features temporelles)
    - Les données historiques du Parquet (pour les lags)

    ENTRÉE  : nom de l'arrêt + date à prédire
    SORTIE  : DataFrame avec les 12 features dans le bon ordre
    """
    if _df_historique is None:
        raise RuntimeError("Données historiques non disponibles")

    date_pred = pd.to_datetime(date_pred)

    # Filtrer l'historique de cet arrêt
    hist_arret = (_df_historique[_df_historique['LIBELLE_ARRET'] == arret]
                  .sort_values('JOUR'))

    if len(hist_arret) == 0:
        raise ValueError(f"Arrêt inconnu : '{arret}'. "
                         f"Vérifiez l'orthographe (majuscules).")

    # ---- Variables temporelles ----
    # On les extrait directement depuis la date demandée
    jour_semaine  = date_pred.dayofweek          # 0=Lundi ... 6=Dimanche
    is_weekend    = int(date_pred.dayofweek >= 5) # 1 si sam/dim
    mois          = date_pred.month               # 1 à 12
    semaine_annee = date_pred.isocalendar()[1]    # numéro de semaine

    # Jours fériés et vacances : lus depuis le calendrier, EXACTEMENT
    # comme preprocessing.py. Une date normalisée (minuit) pour matcher
    # les clés du calendrier.
    date_key = date_pred.normalize()
    is_ferie    = int(date_key in _JOURS_FERIES)
    is_vacances = int(date_key in _JOURS_VACANCES)

    # ---- Variables de lag ----
    # On cherche le trafic de cet arrêt il y a 1 et 7 jours
    date_lag1 = date_pred - pd.Timedelta(days=1)
    date_lag7 = date_pred - pd.Timedelta(days=7)

    def get_nb_vald(hist, target_date):
        """Cherche NB_VALD pour une date donnée, retourne la médiane si absent."""
        row = hist[hist['JOUR'] == target_date]
        if len(row) > 0:
            return float(row['NB_VALD'].values[0])
        return float(hist['NB_VALD'].median())

    nb_vald_lag1 = get_nb_vald(hist_arret, date_lag1)
    nb_vald_lag7 = get_nb_vald(hist_arret, date_lag7)

    # Rolling mean et std sur les 7 derniers jours disponibles
    hist_recente = hist_arret[hist_arret['JOUR'] < date_pred].tail(7)
    rolling_mean = float(hist_recente['NB_VALD'].mean()) if len(hist_recente) > 0 else nb_vald_lag7
    rolling_std  = float(hist_recente['NB_VALD'].std())  if len(hist_recente) > 1 else 0.0

    # ---- Score de congestion ----
    # À l'ENTRAÎNEMENT (preprocessing.py) : NB_VALD DU JOUR / max historique.
    # Il faut reproduire EXACTEMENT ce calcul, sinon le modèle ne peut pas
    # redonner la valeur apprise sur une date passée.
    #
    # DEUX CAS :
    #   - Date PASSÉE (dans les données) : on lit le vrai NB_VALD du jour.
    #     Les features sont identiques à l'entraînement → même prédiction.
    #   - Date FUTURE : le NB_VALD du jour n'existe pas encore (c'est ce
    #     qu'on veut prédire). On l'approxime par la tendance récente
    #     (rolling_mean). C'est une estimation, inévitable pour du futur.
    max_arret = float(hist_arret['NB_VALD'].max())
    ligne_jour = hist_arret[hist_arret['JOUR'] == date_pred]
    if len(ligne_jour) > 0:
        nb_vald_jour = float(ligne_jour['NB_VALD'].values[0])   # date passée : valeur réelle
    else:
        nb_vald_jour = rolling_mean                             # date future : approximation
    score_congestion = round(nb_vald_jour / max_arret, 4) if max_arret > 0 else 0.0

    # Encodage du transporteur
    code_trns = str(hist_arret['CODE_STIF_TRNS'].iloc[0])
    transporteur_encoded = _label_encoder.get(code_trns, 0)

    # ---- Assemblage du vecteur de features ----
    features = {
        'jour_semaine'        : jour_semaine,
        'is_weekend'          : is_weekend,
        'mois'                : mois,
        'semaine_annee'       : semaine_annee,
        'is_ferie'            : is_ferie,
        'is_vacances'         : is_vacances,
        'NB_VALD_lag1'        : nb_vald_lag1,
        'NB_VALD_lag7'        : nb_vald_lag7,
        'rolling_mean_7j'     : rolling_mean,
        'rolling_std_7j'      : rolling_std,
        'score_congestion'    : score_congestion,
        'transporteur_encoded': transporteur_encoded,
    }

    # Retourner dans le bon ordre (celui vu à l'entraînement)
    return pd.DataFrame([features])[_feature_names]


# ============================================================
# FONCTION 1 — Prédiction du trafic
# ============================================================
def predict_traffic(arret: str, date_pred) -> dict:
    """
    Prédit le nombre de validations pour un arrêt à une date donnée.

    ENTRÉE :
        arret     : nom de l'arrêt en majuscules (ex: "CHATELET")
        date_pred : date à prédire (str "2025-10-15" ou objet date)

    SORTIE :
        predicted_validations : nombre de validations prédit
        confidence            : score de confiance (0 à 1)
        features_used         : les features calculées (debug)
        model_used            : nom du modèle utilisé

    EXEMPLE :
        result = predict_traffic("CHATELET", "2025-10-15")
        print(result['predicted_validations'])  # ex: 12 450
    """
    if _traffic_model is None:
        raise RuntimeError(
            "Modèle trafic non disponible. Lancez : python src/train_traffic.py"
        )

    # Normaliser le nom de l'arrêt (les arrêts sont stockés en MAJUSCULES)
    arret = arret.strip().upper()

    # Construire les features
    X = _build_features(arret, date_pred)

    # Prédiction
    prediction = float(_traffic_model.predict(X)[0])
    prediction = max(0, round(prediction))  # pas de valeur négative

    # Score de confiance
    # POUR RANDOM FOREST : on calcule l'écart-type des prédictions
    # de chaque arbre — un petit écart-type = haute confiance
    if hasattr(_traffic_model, 'estimators_'):
        predictions_arbres = np.array([
            tree.predict(X)[0] for tree in _traffic_model.estimators_
        ])
        std_pred   = predictions_arbres.std()
        confidence = round(float(1 / (1 + std_pred / (prediction + 1))), 4)
    else:
        confidence = 0.85  # valeur par défaut si pas de RandomForest

    # ---- Explicabilité SHAP ----
    # POURQUOI : expliquer POURQUOI le modèle a prédit cette valeur.
    # Pour chaque feature, SHAP calcule sa contribution à la prédiction :
    #   contribution positive = pousse la prédiction vers le haut
    #   contribution négative = pousse la prédiction vers le bas
    # On renvoie les 3 features les plus influentes.
    top_factors = []
    if _shap_explainer is not None:
        try:
            shap_values = _shap_explainer.shap_values(X)[0]
            shap_df = pd.DataFrame({
                'feature': _feature_names,
                'shap'   : shap_values,
                'impact' : np.abs(shap_values)
            }).sort_values('impact', ascending=False)

            for _, row in shap_df.head(3).iterrows():
                direction = ('augmente le trafic' if row['shap'] > 0
                             else 'diminue le trafic')
                top_factors.append({
                    'feature'  : row['feature'],
                    'shap'     : round(float(row['shap']), 1),
                    'direction': direction
                })
        except Exception as e:
            logger.warning(f"SHAP non calculé : {e}")

    return {
        'arret'                 : arret,
        'date'                  : str(date_pred),
        'predicted_validations' : int(prediction),
        'confidence'            : confidence,
        'model_used'            : type(_traffic_model).__name__,
        'top_factors'           : top_factors,
        'features_used'         : X.to_dict(orient='records')[0]
    }


# ============================================================
# FONCTION 2 — Détection d'anomalies
# ============================================================
def predict_anomaly(arret: str, nb_validations: int) -> dict:
    """
    Détecte si un nombre de validations est anormal pour un arrêt.

    ENTRÉE :
        arret           : nom de l'arrêt (ex: "GARE DU NORD")
        nb_validations  : nombre de validations observé ce jour

    SORTIE :
        is_anomaly    : True si le trafic est anormal
        anomaly_score : score d'anomalie (plus négatif = plus anormal)
        message       : explication lisible
        z_score       : écart en nombre d'écarts-types

    EXEMPLE :
        result = predict_anomaly("GARE DU NORD", 800)
        print(result['is_anomaly'])  # True si très bas
    """
    if _anomaly_model is None:
        raise RuntimeError(
            "Modèle anomalie non disponible. Lancez : python src/train_anomaly.py"
        )

    if _df_historique is None:
        raise RuntimeError("Données historiques non disponibles")

    # Normaliser le nom de l'arrêt (stocké en MAJUSCULES)
    arret = arret.strip().upper()

    # Historique de l'arrêt
    hist_arret = _df_historique[_df_historique['LIBELLE_ARRET'] == arret]

    if len(hist_arret) == 0:
        raise ValueError(f"Arrêt inconnu : '{arret}'")

    # Z-score : combien d'écarts-types s'éloigne-t-on de la moyenne ?
    # POURQUOI LE Z-SCORE ?
    # Si la moyenne de CHATELET est 10 000 validations avec un écart-type
    # de 2 000, et qu'on observe 800 validations :
    # z = (800 - 10 000) / 2 000 = -4.6
    # Un z-score de -4.6 = très anormal (plus de 4 écarts-types en dessous)
    moyenne = float(hist_arret['NB_VALD'].mean())
    std     = float(hist_arret['NB_VALD'].std())
    z_score = (nb_validations - moyenne) / std if std > 0 else 0.0

    # Features pour Isolation Forest
    ecart_rolling = nb_validations - float(hist_arret['NB_VALD'].tail(7).mean())
    X = pd.DataFrame([{
        'nb_vald_norm' : nb_validations / float(hist_arret['NB_VALD'].max()),
        'ecart_rolling': ecart_rolling,
        'z_score'      : z_score
    }])

    # Prédiction Isolation Forest
    # -1 = anomalie | +1 = normal
    prediction    = int(_anomaly_model.predict(X)[0])
    anomaly_score = float(_anomaly_model.score_samples(X)[0])
    is_anomaly    = prediction == -1

    # Message lisible
    if is_anomaly and z_score < -2:
        message = (f"Trafic anormalement bas — possible grève, incident "
                   f"ou fermeture ({nb_validations:,} vs moyenne {moyenne:,.0f})")
    elif is_anomaly and z_score > 2:
        message = (f"Trafic anormalement élevé — possible événement exceptionnel "
                   f"({nb_validations:,} vs moyenne {moyenne:,.0f})")
    else:
        message = f"Trafic normal ({nb_validations:,} validations)"

    return {
        'arret'         : arret,
        'nb_validations': nb_validations,
        'is_anomaly'    : is_anomaly,
        'anomaly_score' : round(anomaly_score, 4),
        'z_score'       : round(z_score, 2),
        'moyenne_arret' : round(moyenne, 0),
        'message'       : message
    }


# ============================================================
# FONCTION 3 — Clustering des arrêts (versions K=2 et K=5)
# ============================================================
def predict_cluster(arret: str, version: str = "k5") -> dict:
    """
    Retourne le cluster d'appartenance d'un arrêt et ses arrêts similaires.

    ENTRÉE :
        arret   : nom de l'arrêt (ex: "VERSAILLES CH")
        version : "k5" (vue détaillée, 5 profils) ou "k2" (vue simple, 2 groupes)

    SORTIE :
        cluster           : numéro du cluster
        profil            : nom du profil métier (lu depuis le Parquet)
        arrets_similaires : les 5 plus gros arrêts du même cluster
        caracteristiques  : métriques clés de l'arrêt

    EXEMPLE :
        result = predict_cluster("VERSAILLES CH", version="k5")
        print(result['profil'])

    POURQUOI DEUX VERSIONS ?
    - k2 : vue simple (Méga-hubs vs Réseau standard)
    - k5 : vue métier détaillée (5 profils)
    On utilise les deux en production selon le besoin.

    CAS 1 — arrêt connu : on lit directement son cluster dans le
            Parquet arrets_clusters.parquet (instantané).
    CAS 2 — arrêt nouveau : on calcule ses features, on normalise
            avec le scaler, et le modèle prédit son cluster.
    """
    # Choisir le bon modèle et les bonnes colonnes selon la version
    if version == "k2":
        model       = _cluster_model_k2
        col_cluster = "cluster_k2"
        col_profil  = "profil_k2"
    else:  # k5 par défaut
        model       = _cluster_model_k5
        col_cluster = "cluster_k5"
        col_profil  = "profil_k5"

    if model is None:
        raise RuntimeError(
            "Modèle clustering non disponible. Lancez : python src/train_clustering.py"
        )
    if _df_clusters is None:
        raise RuntimeError(
            "Parquet des clusters non disponible. Lancez : python src/train_clustering.py"
        )

    # Normaliser le nom de l'arrêt (stocké en MAJUSCULES)
    arret = arret.strip().upper()

    # ---- CAS 1 : l'arrêt existe dans le Parquet (lecture rapide) ----
    ligne = _df_clusters[_df_clusters['LIBELLE_ARRET'] == arret]

    if len(ligne) > 0:
        cluster = int(ligne[col_cluster].values[0])
        profil  = str(ligne[col_profil].values[0])
        carac = {
            'trafic_moyen_semaine' : round(float(ligne['trafic_moyen_semaine'].values[0]), 0),
            'trafic_moyen_weekend' : round(float(ligne['trafic_moyen_weekend'].values[0]), 0),
            'ratio_weekend_semaine': round(float(ligne['ratio_we_semaine'].values[0]), 2),
            'volume_total_92j'     : round(float(ligne['volume_total'].values[0]), 0)
        }
    else:
        # ---- CAS 2 : arrêt nouveau — le modèle prédit son cluster ----
        if _df_historique is None:
            raise ValueError(f"Arrêt inconnu : '{arret}'")

        hist = _df_historique[_df_historique['LIBELLE_ARRET'] == arret].copy()
        if len(hist) == 0:
            raise ValueError(f"Arrêt inconnu : '{arret}'")

        hist['is_weekend'] = hist['JOUR'].dt.dayofweek >= 5
        ts = float(hist[~hist['is_weekend']]['NB_VALD'].mean())
        tw = float(hist[hist['is_weekend']]['NB_VALD'].mean())
        rw = tw / ts if ts > 0 else 0
        vt = float(hist['NB_VALD'].sum())
        cv = float(hist['NB_VALD'].std() / hist['NB_VALD'].mean()
                   if hist['NB_VALD'].mean() > 0 else 0)

        # Mêmes noms de colonnes qu'à l'entraînement (train_clustering.py)
        X = pd.DataFrame([{
            'trafic_moyen_semaine': ts,
            'trafic_moyen_weekend': tw,
            'ratio_we_semaine'    : rw,
            'volume_total'        : vt,
            'coef_variation'      : cv
        }])

        # Normaliser avec le scaler AVANT de prédire
        # (le modèle a été entraîné sur des données normalisées)
        X_scaled = _cluster_scaler.transform(X) if _cluster_scaler is not None else X
        cluster  = int(model.predict(X_scaled)[0])
        profil   = f"Cluster {cluster}"
        carac = {
            'trafic_moyen_semaine' : round(ts, 0),
            'trafic_moyen_weekend' : round(tw, 0),
            'ratio_weekend_semaine': round(rw, 2),
            'volume_total_92j'     : round(vt, 0)
        }

    # ---- Arrêts similaires : même cluster, lus dans le Parquet ----
    similaires = (_df_clusters[
        (_df_clusters[col_cluster] == cluster) &
        (_df_clusters['LIBELLE_ARRET'] != arret)
    ].nlargest(5, 'volume_total')['LIBELLE_ARRET'].tolist())

    return {
        'arret'            : arret,
        'version'          : version,
        'cluster'          : cluster,
        'profil'           : profil,
        'arrets_similaires': similaires,
        'caracteristiques' : carac
    }


# ============================================================
# FONCTION 4 — Liste de tous les arrêts
# ============================================================
def get_all_arrets() -> list:
    """
    Retourne la liste triée de tous les arrêts disponibles.

    UTILITÉ : remplir les listes déroulantes de Streamlit.
    L'interface a besoin de connaître les 772 arrêts existants.
    """
    if _df_historique is None:
        raise RuntimeError("Données historiques non disponibles")

    arrets = sorted(_df_historique['LIBELLE_ARRET'].unique().tolist())
    return arrets


# ============================================================
# FONCTION 5 — Statistiques d'un arrêt
# ============================================================
def get_arret_stats(arret: str) -> dict:
    """
    Retourne les statistiques historiques d'un arrêt.

    UTILITÉ : afficher un résumé de l'arrêt dans l'interface
    (moyenne, min, max, nombre de jours de données...).
    """
    if _df_historique is None:
        raise RuntimeError("Données historiques non disponibles")

    # Normaliser le nom de l'arrêt (stocké en MAJUSCULES)
    arret = arret.strip().upper()

    hist = _df_historique[_df_historique['LIBELLE_ARRET'] == arret]
    if len(hist) == 0:
        raise ValueError(f"Arrêt inconnu : '{arret}'")

    return {
        'arret'          : arret,
        'nb_jours'       : int(hist['JOUR'].nunique()),
        'trafic_moyen'   : round(float(hist['NB_VALD'].mean()), 0),
        'trafic_median'  : round(float(hist['NB_VALD'].median()), 0),
        'trafic_min'     : int(hist['NB_VALD'].min()),
        'trafic_max'     : int(hist['NB_VALD'].max()),
        'trafic_total'   : int(hist['NB_VALD'].sum()),
        'date_debut'     : str(hist['JOUR'].min().date()),
        'date_fin'       : str(hist['JOUR'].max().date())
    }


# ============================================================
# FONCTION 6 — Liste des clusters avec leurs profils
# ============================================================
def get_all_clusters(version: str = "k5") -> dict:
    """
    Retourne la liste des clusters et leur composition.

    UTILITÉ : afficher la carte des arrêts colorée par cluster,
    ou une vue d'ensemble des profils dans le dashboard.

    version : "k5" (5 profils) ou "k2" (2 groupes)
    """
    if _df_clusters is None:
        raise RuntimeError(
            "Parquet des clusters non disponible. Lancez : python src/train_clustering.py"
        )

    col_cluster = "cluster_k2" if version == "k2" else "cluster_k5"
    col_profil  = "profil_k2"  if version == "k2" else "profil_k5"

    clusters = []
    for cluster_id, grp in _df_clusters.groupby(col_cluster):
        clusters.append({
            'cluster'      : int(cluster_id),
            'profil'       : str(grp[col_profil].iloc[0]),
            'nb_arrets'    : len(grp),
            'volume_moyen' : round(float(grp['volume_total'].mean()), 0),
            'exemples'     : grp.nlargest(5, 'volume_total')['LIBELLE_ARRET'].tolist()
        })

    return {
        'version'     : version,
        'nb_clusters' : len(clusters),
        'clusters'    : clusters
    }


# ============================================================
# FONCTION 7 — Ajouter de nouvelles validations dans MySQL
# ============================================================
def add_validations(data: dict) -> dict:
    """
    Insère une nouvelle ligne de validations dans MySQL.

    POURQUOI DANS MYSQL (le brut) ?
    MySQL est la source de vérité. Le Parquet en découle.
    On ajoute toujours au brut, puis on régénère le Parquet
    (via preprocessing.py) pour mettre à jour les modèles.

    ENTRÉE (dict) : toutes les colonnes de la table validations
        JOUR, CODE_STIF_TRNS, CODE_STIF_RES, CODE_STIF_ARRET,
        ID_ZDC, LIBELLE_ARRET, CATEGORIE_TITRE, NB_VALD

    SORTIE : confirmation de l'insertion
    """
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv

    # OÙ écrire ? Cela dépend des variables d'environnement chargées.
    #   - En LOCAL  : .env → MySQL local, sans SSL
    #   - EN LIGNE  : .env.cloud → Aiven, avec SSL (certificat CA)
    # On détecte le cloud par la présence de DB_SSL_CA. Si le fichier
    # .env.cloud existe, il a priorité (déploiement) ; sinon .env (local).
    if os.path.exists(".env.cloud"):
        load_dotenv(".env.cloud")
    else:
        load_dotenv()

    DB_USER     = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST     = os.getenv("DB_HOST", "localhost")
    DB_PORT     = os.getenv("DB_PORT", "3306")
    DB_NAME     = os.getenv("DB_NAME", "idfm_mobility")
    DB_SSL_CA   = os.getenv("DB_SSL_CA")  # présent seulement pour Aiven

    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # SSL obligatoire pour Aiven, inutile en local. On adapte selon le cas.
    if DB_SSL_CA and os.path.exists(DB_SSL_CA):
        engine = create_engine(url, connect_args={"ssl": {"ca": DB_SSL_CA}})
    else:
        engine = create_engine(url)

    # Construire la ligne à insérer
    ligne = pd.DataFrame([{
        'JOUR'           : data['JOUR'],
        'CODE_STIF_TRNS' : data.get('CODE_STIF_TRNS', '100'),
        'CODE_STIF_RES'  : data.get('CODE_STIF_RES', '110'),
        'CODE_STIF_ARRET': data.get('CODE_STIF_ARRET', '0'),
        'ID_ZDC'         : data.get('ID_ZDC', 0),
        'LIBELLE_ARRET'  : data['LIBELLE_ARRET'],
        'CATEGORIE_TITRE': data['CATEGORIE_TITRE'],
        'NB_VALD'        : int(data['NB_VALD'])
    }])

    # Insérer dans MySQL
    ligne.to_sql('validations', con=engine, if_exists='append', index=False)

    return {
        'status'  : 'success',
        'message' : f"Validation ajoutée : {data['LIBELLE_ARRET']} "
                    f"le {data['JOUR']} ({data['NB_VALD']} validations)",
        'inserted': data
    }


# ============================================================
# FONCTION 8 — Prédiction long terme (Prophet)
# ============================================================
# Deux garde-fous pour Prophet (mêmes seuils que train_prophet.py) :
#   - MIN_JOURS_PROPHET : assez d'historique pour une saison annuelle
#   - MIN_TRAFIC_PROPHET : assez de trafic pour que la prédiction ait un sens.
#     Un arrêt à 105 validations/jour donne un MAPE de 70% ; à 500+ il tient.
MIN_JOURS_PROPHET = 180
MIN_TRAFIC_PROPHET = 500


def predict_long_term(arret: str, date_pred, avec_courbe: bool = False) -> dict:
    """
    Prédit le trafic d'un arrêt à une date LOINTAINE (des mois à l'avance).

    LA DIFFÉRENCE AVEC predict_traffic() :
    predict_traffic utilise lag1 (hier) et lag7 (il y a 7 jours). Pour le
    15 septembre 2026, ces valeurs n'existent pas — il devine avec la médiane.

    Prophet, lui, apprend une équation du temps :
        trafic(t) = tendance(t) + saison_hebdo(t) + saison_annuelle(t) + fériés
    Donnez-lui une date, il l'évalue. Aucun besoin des jours intermédiaires.

    POURQUOI ENTRAÎNER À CHAQUE APPEL ?
    Prophet ne traite qu'une série à la fois : un modèle = un arrêt.
    779 arrêts pré-entraînés, ce serait 779 fichiers .pkl à régénérer à
    chaque nouvelle donnée. Or l'entraînement prend moins d'une seconde
    sur 365 points. On entraîne à la demande, rien à stocker.

    ENTRÉE :
        arret       : nom de l'arrêt (ex: "CHATELET")
        date_pred   : date à prédire (str "2026-09-15" ou objet date)
        avec_courbe : si True, renvoie aussi l'historique et la projection
                      complète — pour tracer un graphique

    SORTIE :
        predicted_validations : la prédiction
        intervalle            : [borne_basse, borne_haute] à 80%
        decomposition         : ce qui compose la prédiction
        modele                : "Prophet"

    EXEMPLE :
        r = predict_long_term("CHATELET", "2026-09-15")
        print(r['predicted_validations'])
    """
    if _df_historique is None:
        raise RuntimeError("Données historiques non disponibles")
    if _prophet_feries is None:
        raise RuntimeError(
            "Calendrier Prophet non disponible. Lancez : python src/train_prophet.py"
        )

    # Import local : Prophet met ~2s à charger (il compile du C++ via Stan).
    # L'importer en tête de fichier ralentirait le démarrage de l'API même
    # quand personne ne demande de prédiction long terme.
    import warnings, logging as _lg
    warnings.filterwarnings("ignore")
    _lg.getLogger('prophet').setLevel(_lg.ERROR)
    _lg.getLogger('cmdstanpy').setLevel(_lg.ERROR)
    from prophet import Prophet

    arret = arret.strip().upper()
    date_pred = pd.to_datetime(date_pred)

    # ---- Historique de l'arrêt, au format Prophet ----
    # Prophet exige exactement deux colonnes : 'ds' (date), 'y' (valeur).
    # Pas de features, pas de lags. Juste le temps et la mesure.
    hist = (_df_historique[_df_historique['LIBELLE_ARRET'] == arret]
            [['JOUR', 'NB_VALD']]
            .rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
            .sort_values('ds'))

    if len(hist) == 0:
        raise ValueError(f"Arrêt inconnu : '{arret}'")

    if len(hist) < MIN_JOURS_PROPHET:
        raise ValueError(
            f"'{arret}' n'a que {len(hist)} jours d'historique "
            f"(minimum {MIN_JOURS_PROPHET}). La saisonnalité annuelle ne peut "
            f"pas être apprise — prédiction long terme indisponible pour cet arrêt."
        )

    if hist['y'].mean() < MIN_TRAFIC_PROPHET:
        raise ValueError(
            f"'{arret}' a un trafic moyen de {hist['y'].mean():.0f} validations/jour "
            f"(minimum {MIN_TRAFIC_PROPHET}). Trop faible pour une prédiction long "
            f"terme fiable — la saisonnalité se noie dans le bruit."
        )

    # ---- Entraînement à la volée (~0.5s) ----
    # seasonality_mode='multiplicative' : le week-end retire un POURCENTAGE
    #   de validations, pas un nombre fixe. Saint-Lazare et un petit arrêt
    #   perdent la même proportion, pas le même nombre.
    # changepoint_prior_scale/range bridés : avec une seule année, Prophet
    #   confondrait le creux de décembre avec un déclin du réseau et
    #   l'extrapolerait sur 2026.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=_prophet_feries,
        seasonality_mode='multiplicative',
        interval_width=0.80,
        changepoint_prior_scale=0.01,
        changepoint_range=0.8,
    )
    model.fit(hist)

    # ---- Prédiction sur la date demandée ----
    fc = model.predict(pd.DataFrame({'ds': [date_pred]})).iloc[0]

    prediction  = max(0, int(round(fc['yhat'])))
    borne_basse = max(0, int(round(fc['yhat_lower'])))
    borne_haute = int(round(fc['yhat_upper']))

    # ---- Décomposition : POURQUOI ce chiffre ? ----
    # C'est l'équivalent du SHAP pour le court terme. On voit ce que chaque
    # composante apporte. Les saisonnalités sont multiplicatives (en %),
    # la tendance est en validations.
    decomposition = {
        'tendance'      : int(round(fc['trend'])),
        'effet_semaine' : round(float(fc['weekly']) * 100, 1),
        'effet_saison'  : round(float(fc['yearly']) * 100, 1),
        'effet_ferie'   : round(float(fc.get('holidays', 0.0)) * 100, 1),
        'jour'          : date_pred.day_name(),
    }

    horizon = (date_pred - hist['ds'].max()).days
    resultat = {
        'arret'                 : arret,
        'date'                  : str(date_pred.date()),
        'predicted_validations' : prediction,
        'intervalle'            : [borne_basse, borne_haute],
        'decomposition'         : decomposition,
        'modele'                : 'Prophet',
        'horizon_jours'         : horizon,
        'jours_historique'      : len(hist),
        'moyenne_historique'    : int(round(hist['y'].mean())),
    }

    # ---- Optionnel : la courbe complète, pour le graphique ----
    if avec_courbe:
        # Projeter 30 jours au-delà de la date demandée, pour que la courbe
        # ne s'arrête pas pile sur le point prédit.
        n_futur = max(horizon + 30, 30)
        futur = model.make_future_dataframe(periods=n_futur)
        courbe = model.predict(futur)

        resultat['courbe'] = {
            'historique': {
                'dates'  : hist['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'valeurs': hist['y'].round().astype(int).tolist(),
            },
            'projection': {
                'dates'       : courbe['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'valeurs'     : courbe['yhat'].round().astype(int).tolist(),
                'borne_basse' : courbe['yhat_lower'].clip(lower=0).round().astype(int).tolist(),
                'borne_haute' : courbe['yhat_upper'].round().astype(int).tolist(),
            }
        }

    return resultat


# ============================================================
# TEST RAPIDE — Point d'entrée
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("TEST DES FONCTIONS DE PRÉDICTION")
    print("=" * 55)

    # Test 1 : prédiction du trafic
    print("\n1. PRÉDICTION DU TRAFIC")
    print("-" * 40)
    try:
        result = predict_traffic("CHATELET", "2026-07-01")  # lendemain des données (futur)
        print(f"Arrêt     : {result['arret']}")
        print(f"Date      : {result['date']}")
        print(f"Prédiction: {result['predicted_validations']:,} validations")
        print(f"Confiance : {result['confidence']:.1%}")
        print(f"Modèle    : {result['model_used']}")
        if result.get('top_factors'):
            print(f"Top 3 facteurs (SHAP) :")
            for fac in result['top_factors']:
                print(f"  {fac['feature']:<20} SHAP={fac['shap']:>10,.1f} -> {fac['direction']}")
    except Exception as e:
        print(f"Erreur : {e}")

    # Test 2 : détection d'anomalie
    print("\n2. DÉTECTION D'ANOMALIE")
    print("-" * 40)
    try:
        result = predict_anomaly("GARE DU NORD", 800)
        print(f"Arrêt          : {result['arret']}")
        print(f"NB validations : {result['nb_validations']:,}")
        print(f"Is anomalie    : {result['is_anomaly']}")
        print(f"Z-score        : {result['z_score']}")
        print(f"Message        : {result['message']}")
    except Exception as e:
        print(f"Erreur (normal si train_anomaly.py pas encore lancé) : {e}")

    # Test 3 : clustering
    print("\n3. CLUSTERING")
    print("-" * 40)
    try:
        result = predict_cluster("VERSAILLES CH", version="k5")
        print(f"Arrêt    : {result['arret']}")
        print(f"Version  : {result['version']}")
        print(f"Cluster  : {result['cluster']}")
        print(f"Profil   : {result['profil']}")
        print(f"Similaires : {result['arrets_similaires']}")

        # Test aussi la version K=2 (vue simple)
        result_k2 = predict_cluster("VERSAILLES CH", version="k2")
        print(f"\n--- Version K=2 (vue simple) ---")
        print(f"Cluster  : {result_k2['cluster']}")
        print(f"Profil   : {result_k2['profil']}")
        print(f"Similaires : {result_k2['arrets_similaires']}")
    except Exception as e:
        print(f"Erreur (normal si train_clustering.py pas encore lancé) : {e}")

    # Test 4 : prédiction long terme (Prophet)
    print("\n4. PRÉDICTION LONG TERME (Prophet)")
    print("-" * 40)
    try:
        result = predict_long_term("CHATELET", "2026-09-15")
        d = result['decomposition']
        print(f"Arrêt        : {result['arret']}")
        print(f"Date         : {result['date']} ({d['jour']})")
        print(f"Horizon      : {result['horizon_jours']} jours après la dernière donnée")
        print(f"Prédiction   : {result['predicted_validations']:,} validations")
        print(f"Intervalle   : {result['intervalle'][0]:,} — {result['intervalle'][1]:,} (80%)")
        print(f"Moyenne hist.: {result['moyenne_historique']:,} validations/jour "
              f"(sur {result['jours_historique']} jours)")
        print(f"Décomposition :")
        print(f"  tendance      {d['tendance']:>10,}")
        print(f"  effet semaine {d['effet_semaine']:>+9.1f}%")
        print(f"  effet saison  {d['effet_saison']:>+9.1f}%")
        print(f"  effet férié   {d['effet_ferie']:>+9.1f}%")
    except Exception as e:
        print(f"Erreur (normal si train_prophet.py pas encore lancé) : {e}")

    print("\n" + "=" * 55)
    print("Tests terminés")
    print("=" * 55)