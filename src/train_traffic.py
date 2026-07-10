# ============================================================
# ÉTAPE 4 — Modèle 1 : Prédiction du trafic IDFM
# Fichier : src/train_traffic.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Le notebook 03_modeling_traffic.ipynb sert à explorer
# et visualiser les modèles. Ce fichier est le script
# de PRODUCTION — il entraîne officiellement les modèles,
# les compare, sélectionne le meilleur et sauvegarde le .pkl.
#
# PRINCIPE DE SÉPARATION DES RESPONSABILITÉS :
# - notebook  → exploration visuelle (pas de sauvegarde)
# - train_traffic.py → entraînement officiel + sauvegarde
# - predict.py       → prédiction en production
#
# ENTRÉE  : data/processed/idfm_features.parquet
# SORTIE  : models/traffic_model.pkl
#           models/feature_names.pkl
#           models/label_encoder.pkl
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
PARQUET_PATH   = "data/processed/idfm_features.parquet"
MODEL_PATH     = "models/traffic_model.pkl"
FEATURES_PATH  = "models/feature_names.pkl"
ENCODER_PATH   = "models/label_encoder.pkl"
EXPLAINER_PATH = "models/shap_explainer.pkl"

# Seuils
OVERFIT_SEUIL  = 0.05   # écart R² train-test max toléré
R2_OBJECTIF    = 0.85   # R² minimum attendu sur le test

# Features utilisées par le modèle
# DOIT être identique à ce qu'on a défini dans le notebook
FEATURES = [
    'jour_semaine',
    'is_weekend',
    'mois',
    'semaine_annee',
    'is_ferie',
    'is_vacances',
    'NB_VALD_lag1',
    'NB_VALD_lag7',
    'rolling_mean_7j',
    'rolling_std_7j',
    'score_congestion',
    'transporteur_encoded'
]
TARGET = 'NB_VALD'


# ============================================================
# FONCTION 1 — Chargement et split temporel
# ============================================================
def load_and_split(parquet_path: str):
    """
    Charge le Parquet et fait le split temporel 80/20.

    POURQUOI UN SPLIT TEMPOREL ?
    On ne peut pas mélanger aléatoirement une série temporelle.
    On ne peut pas utiliser le futur pour prédire le passé.
    80% des jours → TRAIN | 20% des jours → TEST

    POURQUOI PAS DE VALIDATION SET ICI ?
    Contrairement au projet diabète (train/val/test),
    ici on n'a que 92 jours. Ajouter un val set réduirait
    trop le train. On utilise les métriques train/test
    pour détecter l'overfitting.
    """
    logger.info(f"Chargement du Parquet : {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df['JOUR'] = pd.to_datetime(df['JOUR'])
    df = df.sort_values(['LIBELLE_ARRET', 'JOUR']).reset_index(drop=True)

    # Split temporel
    jours        = sorted(df['JOUR'].unique())
    coupure_idx  = int(len(jours) * 0.80)
    date_coupure = jours[coupure_idx]

    train = df[df['JOUR'] <  date_coupure].copy()
    test  = df[df['JOUR'] >= date_coupure].copy()

    X_train = train[FEATURES].copy()
    y_train = train[TARGET].copy()
    X_test  = test[FEATURES].copy()
    y_test  = test[TARGET].copy()

    # Nettoyage des NaN restants
    # POURQUOI : lag7 génère des NaN les 7 premiers jours
    # La régression linéaire plante sur les NaN
    # On remplace par la médiane du TRAIN uniquement
    for col in FEATURES:
        mediane = X_train[col].median()
        X_train[col] = X_train[col].fillna(mediane)
        X_test[col]  = X_test[col].fillna(mediane)

    logger.info(f"Split temporel : {date_coupure}")
    logger.info(f"TRAIN : {len(train):,} lignes | {train['JOUR'].nunique()} jours")
    logger.info(f"TEST  : {len(test):,} lignes  | {test['JOUR'].nunique()} jours")
    logger.info(f"NaN restants : {X_train.isnull().sum().sum()} train | "
                f"{X_test.isnull().sum().sum()} test")

    return X_train, X_test, y_train, y_test, date_coupure


# ============================================================
# FONCTION 2 — Calcul des métriques
# ============================================================
def compute_metrics(y_true, y_pred) -> dict:
    """
    Calcule RMSE, MAE et R² — les 3 métriques de régression.

    RMSE : erreur en nombre de validations (unité = NB_VALD)
           pénalise plus les grandes erreurs
    MAE  : erreur absolue moyenne — plus robuste aux outliers
    R²   : proportion de variance expliquée (objectif > 0.85)
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


# ============================================================
# FONCTION 3 — Statut overfitting
# ============================================================
def overfitting_status(r2_train: float, r2_test: float) -> str:
    """
    Détermine si le modèle overfitte, underfitte ou est OK.

    OVERFITTING  : R² train >> R² test (modèle mémorise le train)
    UNDERFITTING : R² train ET test tous les deux bas
    OK           : R² train et test proches ET élevés
    """
    ecart = r2_train - r2_test
    if ecart > OVERFIT_SEUIL:
        return f"OVERFITTING (ecart={ecart:.3f} > {OVERFIT_SEUIL})"
    elif r2_test < 0.5:
        return f"UNDERFITTING (R² test={r2_test:.3f} < 0.5)"
    else:
        return f"OK (ecart={ecart:.3f})"


# ============================================================
# FONCTION 4 — Entraînement de tous les modèles avec MLflow
# ============================================================
def train_models(X_train, X_test, y_train, y_test):
    """
    Entraîne les 4 modèles, les compare et sélectionne le meilleur.

    POURQUOI MLFLOW ?
    MLflow trace automatiquement chaque expérience :
    - Les hyperparamètres (n_estimators, max_depth...)
    - Les métriques (RMSE, MAE, R²)
    - Les modèles (.pkl)
    On peut comparer toutes les expériences dans l'UI MLflow.

    CRITÈRES DE SÉLECTION :
    1. Statut = OK (pas d'overfitting)
    2. R² test maximum
    3. RMSE test minimum
    """
    # Définition des 4 modèles
    models = {
        "LinearRegression": LinearRegression(),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

    # Configuration MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
    mlflow.set_experiment("idfm-traffic-prediction")

    best_r2    = -np.inf
    best_model = None
    best_name  = ""
    resultats  = []

    for name, model in models.items():
        logger.info(f"Entraînement : {name}...")

        with mlflow.start_run(run_name=name):

            # Entraînement
            model.fit(X_train, y_train)

            # Prédictions
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)

            # Métriques
            m_train  = compute_metrics(y_train, y_pred_train)
            m_test   = compute_metrics(y_test,  y_pred_test)
            statut   = overfitting_status(m_train['R2'], m_test['R2'])

            # Logging MLflow — paramètres
            mlflow.log_param("model",      name)
            mlflow.log_param("n_features", len(FEATURES))

            # Hyperparamètres selon le modèle
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, 'max_depth') and model.max_depth:
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, 'learning_rate'):
                mlflow.log_param("learning_rate", model.learning_rate)

            # Logging MLflow — métriques train
            mlflow.log_metric("rmse_train", m_train['RMSE'])
            mlflow.log_metric("mae_train",  m_train['MAE'])
            mlflow.log_metric("r2_train",   m_train['R2'])

            # Logging MLflow — métriques test
            mlflow.log_metric("rmse_test",  m_test['RMSE'])
            mlflow.log_metric("mae_test",   m_test['MAE'])
            mlflow.log_metric("r2_test",    m_test['R2'])

            # Écart overfitting
            mlflow.log_metric("ecart_r2",   m_train['R2'] - m_test['R2'])

            # Affichage
            print(f"\n{'='*55}")
            print(f"MODÈLE : {name}")
            print(f"{'='*55}")
            print(f"  RMSE  : Train={m_train['RMSE']:>10,.1f} | "
                  f"Test={m_test['RMSE']:>10,.1f}")
            print(f"  MAE   : Train={m_train['MAE']:>10,.1f}  | "
                  f"Test={m_test['MAE']:>10,.1f}")
            print(f"  R²    : Train={m_train['R2']:>10.4f} | "
                  f"Test={m_test['R2']:>10.4f}")
            print(f"  Statut: {statut}")

            # Sélection du meilleur modèle
            # Critère : pas d'overfitting ET R² test maximum
            if statut.startswith("OK") and m_test['R2'] > best_r2:
                best_r2    = m_test['R2']
                best_model = model
                best_name  = name

            resultats.append({
                'Modèle'    : name,
                'RMSE_train': m_train['RMSE'],
                'RMSE_test' : m_test['RMSE'],
                'MAE_test'  : m_test['MAE'],
                'R2_train'  : m_train['R2'],
                'R2_test'   : m_test['R2'],
                'Ecart_R2'  : m_train['R2'] - m_test['R2'],
                'Statut'    : statut
            })

    # Résumé comparatif
    df_res = pd.DataFrame(resultats)
    print(f"\n{'='*55}")
    print("COMPARAISON FINALE")
    print(f"{'='*55}")
    print(df_res[['Modèle','RMSE_test','R2_test','Statut']].to_string(index=False))
    print(f"\nMeilleur modèle auto : {best_name} (R²={best_r2:.4f})")

    return best_model, best_name, df_res


# ============================================================
# FONCTION 5 — Sauvegarde des artefacts
# ============================================================
def save_artifacts(model, model_name: str) -> None:
    """
    Sauvegarde le modèle et les métadonnées nécessaires à l'API.

    FICHIERS SAUVEGARDÉS :
    - traffic_model.pkl  : le modèle entraîné
    - feature_names.pkl  : liste des features dans le bon ordre
    - label_encoder.pkl  : mapping transporteur_encoded

    POURQUOI CES 3 FICHIERS ?
    L'API FastAPI doit pouvoir recréer exactement la même
    transformation qu'à l'entraînement pour chaque prédiction.
    Sans feature_names.pkl, elle ne saurait pas l'ordre des features.
    Sans label_encoder.pkl, elle ne pourrait pas encoder le transporteur.
    """
    os.makedirs("models", exist_ok=True)

    # Modèle
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    # Noms des features
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(FEATURES, f)
    logger.info(f"Features sauvegardées : {FEATURES_PATH}")

    # Label encoder transporteur
    # On recrée le mapping depuis le Parquet
    df = pd.read_parquet(PARQUET_PATH)
    codes   = df['CODE_STIF_TRNS'].astype(str).unique()
    mapping = {code: i for i, code in enumerate(sorted(codes))}
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(mapping, f)
    logger.info(f"Label encoder sauvegardé : {ENCODER_PATH}")

    # Vérification : recharger et tester
    with open(MODEL_PATH, 'rb') as f:
        model_check = pickle.load(f)
    logger.info(f"Vérification OK — modèle rechargé : {type(model_check).__name__}")

    print(f"\nFichiers sauvegardés :")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {FEATURES_PATH}")
    print(f"  -> {ENCODER_PATH}")
    
# ============================================================
# FONCTION 6 — Création et sauvegarde de l'explainer SHAP
# ============================================================
def save_shap_explainer(model, X_train) -> None:
    """
    Crée un explainer SHAP sur le modèle et le sauvegarde.

    POURQUOI SHAP ?
    SHAP explique POURQUOI le modèle fait chaque prédiction.
    Pour chaque feature, il calcule sa contribution à la prédiction :
    - contribution positive = pousse la prédiction vers le haut
    - contribution négative = pousse la prédiction vers le bas

    Résultat attendu : lag7 et is_weekend seront les plus importants.

    POURQUOI TreeExplainer ?
    Notre meilleur modèle est RandomForest (basé sur des arbres).
    TreeExplainer est optimisé pour les modèles à base d'arbres —
    rapide et exact.
    """
    logger.info("Création de l'explainer SHAP...")

    # TreeExplainer fonctionne pour RandomForest, XGBoost, DecisionTree
    if hasattr(model, 'estimators_') or 'XGB' in type(model).__name__:
        explainer = shap.TreeExplainer(model)
    else:
        # Pour LinearRegression, on utilise un explainer générique
        explainer = shap.Explainer(model, X_train)

    # Sauvegarde
    with open(EXPLAINER_PATH, 'wb') as f:
        pickle.dump(explainer, f)
    logger.info(f"Explainer SHAP sauvegardé : {EXPLAINER_PATH}")

    # Vérification : calculer les valeurs SHAP sur un échantillon
    sample = X_train.iloc[:100]
    shap_values = explainer.shap_values(sample)

    # Importance globale = moyenne des |valeurs SHAP|
    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        'feature': FEATURES,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*55}")
    print("IMPORTANCE DES FEATURES (SHAP)")
    print(f"{'='*55}")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:<22} : {row['importance']:>12,.1f}")

    print(f"\n  -> {EXPLAINER_PATH}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def train():
    """
    Lance le pipeline complet d'entraînement.

    ORDRE :
    1. Charger le Parquet + split temporel
    2. Entraîner les 4 modèles avec MLflow
    3. Sélectionner le meilleur (R² test max sans overfitting)
    4. Sauvegarder le modèle + artefacts dans models/
    """
    logger.info("=" * 60)
    logger.info("DÉMARRAGE — Entraînement modèle prédiction trafic")
    logger.info("=" * 60)

    # Étape 1 : chargement et split
    X_train, X_test, y_train, y_test, date_coupure = load_and_split(PARQUET_PATH)

    # Étape 2 : entraînement + comparaison
    best_model, best_name, resultats = train_models(
        X_train, X_test, y_train, y_test
    )

    # Étape 3 : sauvegarde
    if best_model is not None:
        save_artifacts(best_model, best_name)
        # Étape 4 : créer l'explainer SHAP
        save_shap_explainer(best_model, X_train)
    else:
        logger.error("Aucun modèle valide trouvé — vérifiez les données")
        return

    logger.info("=" * 60)
    logger.info(f"TERMINÉ — Meilleur modèle : {best_name}")
    logger.info("=" * 60)

    print(f"\n{'='*55}")
    print(f"RÉSUMÉ FINAL")
    print(f"{'='*55}")
    print(f"Meilleur modèle sauvegardé : {best_name}")
    print(f"Chemin                     : {MODEL_PATH}")
    print(f"Lancer MLflow UI avec      : mlflow ui")
    print(f"{'='*55}")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    train()