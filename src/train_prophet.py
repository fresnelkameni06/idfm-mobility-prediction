# ============================================================
# ÉTAPE 4bis — Modèle 4 : Prédiction long terme (Prophet)
# Fichier : src/train_prophet.py
# ============================================================
#
# À QUOI SERT PROPHET ?
# Le Random Forest a besoin du trafic d'hier (lag1) et d'il y a 7 jours
# (lag7). Pour prédire le 15 septembre 2026, ces valeurs n'existent pas.
# Il devine avec la médiane, et se trompe.
#
# Prophet ne regarde jamais hier. Il apprend une ÉQUATION DU TEMPS :
#     trafic(t) = tendance(t) + saison_hebdo(t) + saison_annuelle(t) + fériés
#
# Donnez-lui une date, il l'évalue. Point. Pas besoin des jours
# intermédiaires. C'est exactement ce qu'il vous faut : choisir un
# arrêt, choisir une date lointaine, obtenir la prédiction.
#
# ------------------------------------------------------------
# POURQUOI CE FICHIER N'ENTRAÎNE PAS DE MODÈLE ?
#
# Prophet ne traite qu'UNE série temporelle à la fois. Un modèle
# Prophet = un arrêt. Pour 779 arrêts, il faudrait 779 fichiers .pkl,
# à régénérer à chaque nouvelle donnée.
#
# Or Prophet s'entraîne en 0,6 seconde sur 365 points (contre 30s pour
# le Random Forest sur 278 000 lignes). On l'entraîne donc À LA VOLÉE,
# au moment de la requête, dans predict.py. Rien à stocker, toujours
# à jour.
#
# Ce script fait donc deux choses :
#   1. Prépare le calendrier des fériés au format Prophet → prophet_feries.pkl
#      (pour ne pas le reconstruire à chaque prédiction)
#   2. VALIDE que l'approche tient sur vos données : entraîne sur
#      quelques arrêts, mesure l'erreur, vérifie que 2026 est plausible
#
# LANCER :
#   python src/train_prophet.py
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import time
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(message)s')
logger = logging.getLogger(__name__)

# Prophet compile du C++ via Stan et log abondamment. On le fait taire.
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

from prophet import Prophet

# ============================================================
# CHEMINS
# ============================================================
PARQUET_PATH    = "data/processed/idfm_features.parquet"
CALENDRIER_PATH = "data/external/calendrier_feries_fr.csv"
FERIES_PATH     = "models/prophet_feries.pkl"

# DEUX garde-fous, pour deux problèmes différents :
#
# MIN_JOURS — un arrêt apparu en cours d'année n'a pas assez d'historique
# pour qu'une saisonnalité annuelle soit apprise. Prophet rendrait du bruit.
MIN_JOURS = 180
#
# MIN_TRAFIC — un arrêt à 9 validations/jour est du bruit pur. Prédire 4 ou
# 15, c'est une erreur de 66% en pourcentage pour 3 validations d'écart. La
# saisonnalité n'a aucun sens à cette échelle. On refuse aussi ces arrêts :
# Prophet est fait pour les gros pôles (gares, hubs), pas pour un arrêt
# désert. En dessous de cette moyenne journalière, prédiction indisponible.
#
# Pourquoi 500 et non 100 ? Les mesures le montrent : un arrêt à ~100/jour
# donne 70% de MAPE, un arrêt à ~5000/jour donne 13%. La rupture se situe
# vers 500. En dessous, la saisonnalité se noie dans le bruit journalier.
MIN_TRAFIC = 500

TEST_DAYS = 30   # split temporel pour la validation


# ============================================================
# FONCTION 1 — Le calendrier au format Prophet
# ============================================================
def build_holidays(calendrier_path: str = CALENDRIER_PATH) -> pd.DataFrame:
    """
    Convertit notre calendrier au format que Prophet attend.

    POURQUOI DONNER LES FÉRIÉS À PROPHET ?
    Sans eux, il verrait le 14 juillet comme un lundi ordinaire et
    s'étonnerait de la chute de trafic. En les déclarant, il apprend
    "un jour férié fait chuter le trafic de X%" — puis applique cet
    effet au 14 juillet 2026, une date qu'il n'a jamais vue.

    FORMAT : colonnes 'holiday' (nom du groupe) et 'ds' (date).

    POURQUOI SEULEMENT LES FÉRIÉS, PAS LES VACANCES ?
    Un férié est un choc ponctuel : un jour, une chute. Prophet excelle
    là-dessus. Les vacances scolaires durent des semaines — c'est une
    saison, pas un événement. La saisonnalité annuelle de Prophet les
    capture déjà (le creux d'août est dans la courbe). Les déclarer
    comme "holidays" reviendrait à modéliser 250 jours d'anomalies
    sur 730, ce qui brouille le modèle.
    """
    if not os.path.exists(calendrier_path):
        logger.warning(f"Calendrier absent ({calendrier_path}) — "
                       f"Prophet n'apprendra pas l'effet des fériés")
        return None

    cal = pd.read_csv(calendrier_path)
    cal['date'] = pd.to_datetime(cal['date'])

    holidays = pd.DataFrame({
        'holiday': 'ferie',
        'ds': cal.loc[cal['type'] == 'ferie', 'date'],
        'lower_window': 0,
        'upper_window': 0,
    })

    n_2026 = (holidays['ds'].dt.year == 2026).sum()
    logger.info(f"Fériés : {len(holidays)} jours, dont {n_2026} en 2026 "
                f"(indispensable pour prédire au-delà de décembre 2025)")
    return holidays


# ============================================================
# FONCTION 2 — Construire un modèle Prophet configuré
# ============================================================
def make_model(holidays: pd.DataFrame) -> Prophet:
    """
    Crée un Prophet réglé pour du trafic de transport journalier.

    LES CHOIX, ET POURQUOI :

    yearly_seasonality=True
        Le cœur du sujet. C'est ce qui capture le creux d'août et le
        pic de septembre. Sans une année complète de données, ce terme
        n'aurait rien à apprendre — d'où l'ingestion des 4 trimestres.

    weekly_seasonality=True
        Le cycle lundi→dimanche, le signal le plus fort du trafic urbain.

    daily_seasonality=False
        Nos données sont journalières. Pas d'heures à modéliser.

    seasonality_mode='multiplicative'
        Le mode par défaut ('additive') suppose que le week-end retire
        un nombre FIXE de validations. Or il retire un POURCENTAGE :
        Saint-Lazare perd plus de validations qu'un petit arrêt, mais
        la même proportion. 'multiplicative' modélise ça correctement.

    changepoint_prior_scale=0.01  (défaut : 0.05)
    changepoint_range=0.8
        LE PIÈGE À CONNAÎTRE. Prophet cherche des "points de rupture" où
        la tendance change. Avec UNE SEULE année, il ne peut distinguer
        le creux saisonnier de décembre d'un déclin structurel du réseau.
        Il risque d'extrapoler la chute de Noël indéfiniment et de
        prédire un arrêt moribond en 2026.

        Ces réglages le brident : tendance plus rigide, et aucune rupture
        dans les 20% finaux de la série, là où Prophet manque de recul.
    """
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative',
        interval_width=0.80,
        changepoint_prior_scale=0.01,
        changepoint_range=0.8,
    )


# ============================================================
# FONCTION 3 — Entraîner Prophet sur UN arrêt
# ============================================================
def fit_arret(df: pd.DataFrame, arret: str, holidays: pd.DataFrame) -> Prophet:
    """
    Entraîne un Prophet sur l'historique d'un seul arrêt.

    Prophet exige exactement deux colonnes : 'ds' (date) et 'y' (valeur).
    Pas de features, pas de lags. Juste le temps et la mesure.
    C'est toute la différence avec un modèle tabulaire.
    """
    hist = (df[df['LIBELLE_ARRET'] == arret][['JOUR', 'NB_VALD']]
            .rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
            .sort_values('ds'))

    if len(hist) < MIN_JOURS:
        raise ValueError(
            f"'{arret}' n'a que {len(hist)} jours d'historique "
            f"(minimum {MIN_JOURS}). La saisonnalité annuelle ne peut "
            f"pas être apprise de façon fiable."
        )

    if hist['y'].mean() < MIN_TRAFIC:
        raise ValueError(
            f"'{arret}' n'a qu'un trafic moyen de {hist['y'].mean():.0f} "
            f"validations/jour (minimum {MIN_TRAFIC}). Trop faible pour une "
            f"prédiction fiable — la saisonnalité se noie dans le bruit."
        )

    model = make_model(holidays)
    model.fit(hist)
    return model


# ============================================================
# FONCTION 4 — Valider sur un arrêt (split temporel)
# ============================================================
def valider_arret(df: pd.DataFrame, arret: str, holidays: pd.DataFrame) -> dict:
    """
    Entraîne sur tout sauf les 30 derniers jours, puis prédit ces 30 jours.

    POURQUOI UN SPLIT TEMPOREL ET NON ALÉATOIRE ?
    Un split aléatoire laisserait Prophet voir des jours de décembre
    pendant l'entraînement, puis lui demanderait de prédire novembre.
    Il "connaîtrait le futur". Le split temporel reproduit la réalité :
    on apprend le passé, on prédit l'avenir.

    MAPE : erreur en pourcentage. Comparable entre arrêts de tailles
    différentes, contrairement au MAE en validations brutes.
    """
    hist = (df[df['LIBELLE_ARRET'] == arret][['JOUR', 'NB_VALD']]
            .rename(columns={'JOUR': 'ds', 'NB_VALD': 'y'})
            .sort_values('ds'))
    if len(hist) < MIN_JOURS or hist['y'].mean() < MIN_TRAFIC:
        return None

    train, test = hist.iloc[:-TEST_DAYS], hist.iloc[-TEST_DAYS:]

    t0 = time.time()
    m = make_model(holidays)
    m.fit(train)
    duree = time.time() - t0

    fc = m.predict(test[['ds']])
    y_true, y_pred = test['y'].values, fc['yhat'].values

    # Éviter la division par zéro sur les jours sans trafic
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    # Le modèle final, réentraîné sur TOUT (y compris décembre)
    m_final = make_model(holidays)
    m_final.fit(hist)
    fut = pd.DataFrame({'ds': pd.to_datetime(['2026-03-15', '2026-09-15'])})
    proj = m_final.predict(fut)

    return {
        'arret': arret,
        'jours': len(hist),
        'moyenne': float(hist['y'].mean()),
        'MAE': mae,
        'MAPE': mape,
        'duree_fit': duree,
        'proj_mars': float(proj.iloc[0]['yhat']),
        'proj_sept': float(proj.iloc[1]['yhat']),
    }


# ============================================================
# PIPELINE — Préparation + validation
# ============================================================
def run():
    logger.info("=" * 64)
    logger.info("MODÈLE 4 — Prophet (prédiction long terme, par arrêt)")
    logger.info("=" * 64)

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"{PARQUET_PATH} introuvable. Lancez : python src/preprocessing.py"
        )

    df = pd.read_parquet(PARQUET_PATH)
    df['JOUR'] = pd.to_datetime(df['JOUR'])
    logger.info(f"Parquet : {len(df):,} lignes | {df['JOUR'].nunique()} jours | "
                f"{df['LIBELLE_ARRET'].nunique()} arrêts")

    # ---- Combien d'arrêts sont utilisables ? ----
    # Deux critères : assez de jours ET assez de trafic.
    jours_par_arret   = df.groupby('LIBELLE_ARRET')['JOUR'].nunique()
    trafic_par_arret  = df.groupby('LIBELLE_ARRET')['NB_VALD'].mean()

    assez_jours  = jours_par_arret  >= MIN_JOURS
    assez_trafic = trafic_par_arret >= MIN_TRAFIC
    utilisables  = (assez_jours & assez_trafic).sum()
    exclus_jours  = (~assez_jours).sum()
    exclus_trafic = (assez_jours & ~assez_trafic).sum()

    print("\n" + "=" * 64)
    print("COUVERTURE DES ARRÊTS")
    print("=" * 64)
    print(f"  Historique médian      : {int(jours_par_arret.median())} jours")
    print(f"  Trafic médian          : {int(trafic_par_arret.median())} validations/jour")
    print(f"  Utilisables            : {utilisables} arrêts")
    print(f"    (≥{MIN_JOURS}j d'historique ET ≥{MIN_TRAFIC} validations/jour)")
    if exclus_jours:
        print(f"  Exclus — trop récents  : {exclus_jours} arrêts (<{MIN_JOURS}j)")
    if exclus_trafic:
        print(f"  Exclus — trafic faible : {exclus_trafic} arrêts (<{MIN_TRAFIC}/j)")
        print(f"    → petits arrêts où la prédiction n'aurait pas de sens")
    print(f"  predict_long_term() refusera les arrêts exclus explicitement")
    print("=" * 64)

    # ---- Sauvegarder le calendrier pour predict.py ----
    holidays = build_holidays()
    os.makedirs("models", exist_ok=True)
    with open(FERIES_PATH, 'wb') as f:
        pickle.dump(holidays, f)
    logger.info(f"Sauvegardé : {FERIES_PATH} (lu par predict.py à chaque prédiction)")

    # ---- Valider sur un échantillon d'arrêts RÉELLEMENT servis ----
    # On ne valide que sur des arrêts qui passent les deux seuils : inutile
    # de mesurer un MAPE de 66% sur un arrêt à 9 validations qu'on refusera
    # de toute façon. On prend un gros, un moyen, un petit — parmi les valides.
    volumes = df.groupby('LIBELLE_ARRET')['NB_VALD'].sum().sort_values(ascending=False)
    valides = [a for a in volumes.index
               if jours_par_arret[a] >= MIN_JOURS and trafic_par_arret[a] >= MIN_TRAFIC]
    echantillon = [valides[0], valides[len(valides)//2], valides[-1]]

    logger.info(f"Validation sur 3 arrêts servis (gros / moyen / petit)...")
    resultats = [r for a in echantillon if (r := valider_arret(df, a, holidays))]

    print("\n" + "=" * 64)
    print("VALIDATION — test sur les 30 derniers jours de 2025")
    print("=" * 64)
    print(f"{'ARRÊT':<20} {'MOY/j':>9} {'MAE':>9} {'MAPE':>7} {'FIT':>6}")
    print("-" * 64)
    for r in resultats:
        print(f"{r['arret'][:20]:<20} {r['moyenne']:>9,.0f} {r['MAE']:>9,.0f} "
              f"{r['MAPE']:>6.1f}% {r['duree_fit']:>5.1f}s")
    print("=" * 64)

    # MÉDIAN et non moyen : sur 3 arrêts, un seul cas difficile (le plus
    # petit, toujours près du seuil) tirerait la moyenne vers le haut et
    # donnerait une image faussement pessimiste. La médiane est robuste.
    mape_median = float(np.median([r['MAPE'] for r in resultats]))
    fit_moyen   = np.mean([r['duree_fit'] for r in resultats])

    # Repère : 10-20% est normal pour du trafic journalier réel, qui reste
    # bruité (météo, incidents, événements ponctuels). Les gros pôles
    # tournent autour de 15-20%, les arrêts moyens sont meilleurs.
    if mape_median < 20:
        logger.info(f"MAPE médian {mape_median:.1f}% — saisonnalité bien capturée")
    else:
        logger.warning(f"MAPE médian {mape_median:.1f}% — vérifiez la période de "
                       f"test (grèves ? événements exceptionnels ?)")
    logger.info(f"Entraînement : {fit_moyen:.1f}s par arrêt "
                f"→ acceptable pour un entraînement à la volée")

    # ---- Les projections 2026 tiennent-elles debout ? ----
    print("\nPROJECTIONS 2026 — restent-elles dans l'ordre de grandeur ?")
    print("-" * 64)
    print(f"{'ARRÊT':<20} {'MOY 2025':>10} {'15/03/26':>10} {'15/09/26':>10}")
    print("-" * 64)
    suspect = False
    for r in resultats:
        ratio = min(r['proj_mars'], r['proj_sept']) / r['moyenne']
        flag = "  <-- suspect" if ratio < 0.5 else ""
        print(f"{r['arret'][:20]:<20} {r['moyenne']:>10,.0f} "
              f"{r['proj_mars']:>10,.0f} {r['proj_sept']:>10,.0f}{flag}")
        if ratio < 0.5:
            suspect = True
    print("-" * 64)

    if suspect:
        print("\nATTENTION — une projection tombe sous la moitié de la moyenne.")
        print("Prophet a probablement pris le creux de fin d'année pour un")
        print("déclin de fond. Avec une seule année, il ne peut pas faire la")
        print("différence. Les garde-fous (changepoint_range) atténuent sans")
        print("supprimer. À mentionner si vous présentez ce modèle.")

    print("\nCE QUE PROPHET SAIT, ET NE SAIT PAS")
    print("-" * 64)
    print("Il connaît le calendrier : jour de semaine, mois, jours fériés.")
    print("Il ignore l'actualité : une grève hier ne changera rien à sa")
    print("prédiction pour demain. C'est le rôle du Random Forest.")
    print()
    print("Il n'a vu qu'UNE année. Il n'a jamais observé deux mois de mars")
    print("pour savoir à quoi ressemble un mars typique. Ses projections")
    print("au-delà de décembre 2025 sont un ordre de grandeur, pas un chiffre.")
    print("-" * 64)


if __name__ == "__main__":
    run()
    logger.info("=" * 64)
    logger.info("TERMINÉ — Prochaine étape : predict_long_term() dans predict.py")
    logger.info("=" * 64)