# ============================================================
# ÉTAPE 8 — Interface Streamlit — IDFM Mobility
# Fichier : streamlit_app/app.py
# ============================================================
#
# POURQUOI CE FICHIER ?
# Interface web interactive qui consomme l'API FastAPI.
# L'utilisateur prédit le trafic (court ET long terme), détecte
# des anomalies, explore les clusters — sans écrire de code.
#
# ARCHITECTURE :
#   Streamlit → requêtes HTTP → API FastAPI → modèles ML
#
# LANCER (l'API doit tourner en parallèle) :
#   Terminal 1 : uvicorn api.main:app --reload
#   Terminal 2 : streamlit run streamlit_app/app.py
# ============================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
# Adresse de l'API. Configurable via variable d'environnement :
#   - En LOCAL sans Docker : http://localhost:8000 (défaut)
#   - Entre CONTENEURS      : http://api:8000 (nom du service compose)
#   - EN LIGNE              : l'URL publique de l'API déployée
# "localhost" dans un conteneur = le conteneur lui-même, PAS l'API.
# D'où la nécessité de rendre cette adresse configurable.
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="IDFM Mobility",
    page_icon="metro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE — palette sobre et professionnelle
# ============================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { color: #1a2e44; font-weight: 600; }
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #2563a8; }
    .metric-label { font-size: 0.85rem; color: #64748b; margin-top: 0.3rem; }
    .stButton>button {
        background: #2563a8; color: white; border: none;
        border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 500;
    }
    .stButton>button:hover { background: #1a4a80; }
    div[data-testid="stSidebarNav"] { display: none; }
    .info-band {
        background: #eef4fb; border-left: 3px solid #2563a8;
        border-radius: 6px; padding: 0.8rem 1rem; font-size: 0.88rem;
        color: #33475b; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FONCTIONS UTILITAIRES — appels API
# ============================================================
@st.cache_data(ttl=300)
def api_get(endpoint: str):
    """Appel GET à l'API avec gestion d'erreur."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None

def api_post(endpoint: str, payload: dict, timeout: int = 30):
    """Appel POST à l'API avec gestion d'erreur.

    timeout à 30s : la prédiction long terme entraîne Prophet à la volée,
    ce qui prend quelques secondes (contre une réponse instantanée pour
    les autres endpoints)."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        return r.status_code, r.json()
    except requests.exceptions.RequestException as e:
        return None, {"detail": str(e)}

@st.cache_data(ttl=300)
def get_arrets_list():
    """Récupère la liste des arrêts pour les menus déroulants."""
    data = api_get("/arrets")
    return data["arrets"] if data else []

@st.cache_data(ttl=300)
def get_data_range():
    """Renvoie (date_debut, date_fin) des données, lues sur un gros arrêt.

    POURQUOI : l'interface ne doit RIEN coder en dur sur la période
    (ni "92 jours", ni "1er juillet"). L'utilisateur ajoutera des données
    progressivement — l'interface doit refléter la plage réelle du moment.
    On interroge un arrêt à fort trafic, présent sur toute la période."""
    for a in ["CHATELET", "SAINT-LAZARE", "GARE DU NORD"]:
        s = api_get(f"/stats/{a}")
        if s:
            return s["date_debut"], s["date_fin"], s["nb_jours"]
    return None, None, None


# ============================================================
# VÉRIFICATION DE LA CONNEXION API
# ============================================================
health = api_get("/health")
api_online = health is not None

# ============================================================
# BARRE LATÉRALE — Navigation
# ============================================================
with st.sidebar:
    st.markdown("### Analyse de mobilité")
    st.markdown("Île-de-France Mobilités")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Vue d'ensemble", "Prédiction court terme", "Prédiction long terme",
         "Détection d'anomalies", "Profils d'arrêts", "Ajouter des données"],
        label_visibility="collapsed"
    )

    st.divider()
    if api_online:
        st.success("API connectée")
        models = health.get("models", {})
        st.caption(f"Trafic : {models.get('traffic', '?')}")
        st.caption(f"Long terme : {models.get('prophet', '?')}")
        st.caption(f"Anomalies : {models.get('anomaly', '?')}")
        st.caption(f"Clustering : {models.get('clustering', '?')}")
    else:
        st.error("API hors ligne")
        st.caption("Lancez : uvicorn api.main:app --reload")


# ============================================================
# GARDE — si l'API est hors ligne
# ============================================================
if not api_online:
    st.title("Analyse des flux de mobilité — Île-de-France")
    st.warning(
        "L'API n'est pas accessible. Démarrez-la dans un terminal séparé "
        "avec la commande : `uvicorn api.main:app --reload`, puis rechargez cette page."
    )
    st.stop()

arrets = get_arrets_list()
DATE_DEBUT, DATE_FIN, NB_JOURS = get_data_range()

# Conversion en objets date pour les widgets
def _to_date(s, fallback):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return fallback
d_fin  = _to_date(DATE_FIN, date.today())
d_debut = _to_date(DATE_DEBUT, date(2025, 1, 1))


# ============================================================
# PAGE 1 — VUE D'ENSEMBLE
# ============================================================
if page == "Vue d'ensemble":
    st.title("Analyse des flux de mobilité — Île-de-France")
    st.markdown("Prédiction du trafic, détection d'anomalies et segmentation "
                "des arrêts du réseau de transport francilien.")
    st.divider()

    clusters_data = api_get("/clusters?version=k5")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(arrets)}</div>'
                    f'<div class="metric-label">Arrêts analysés</div></div>',
                    unsafe_allow_html=True)
    with col2:
        nb_clusters = clusters_data["nb_clusters"] if clusters_data else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nb_clusters}</div>'
                    f'<div class="metric-label">Profils identifiés</div></div>',
                    unsafe_allow_html=True)
    with col3:
        # Nombre de jours RÉEL, lu depuis l'API — pas de valeur figée
        jours_txt = f"{NB_JOURS}" if NB_JOURS else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{jours_txt}</div>'
                    f'<div class="metric-label">Jours de données</div></div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">4</div>'
                    f'<div class="metric-label">Modèles ML</div></div>',
                    unsafe_allow_html=True)

    if DATE_DEBUT and DATE_FIN:
        st.caption(f"Période couverte : du {DATE_DEBUT} au {DATE_FIN}")

    st.divider()

    if clusters_data:
        st.subheader("Répartition des arrêts par profil")
        df_clusters = pd.DataFrame(clusters_data["clusters"])

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = px.bar(
                df_clusters.sort_values("nb_arrets"),
                x="nb_arrets", y="profil", orientation="h",
                color="nb_arrets", color_continuous_scale="Blues",
                labels={"nb_arrets": "Nombre d'arrêts", "profil": ""}
            )
            fig.update_layout(showlegend=False, height=350,
                              coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.pie(
                df_clusters, values="nb_arrets", names="profil",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=-0.1))
            fig2.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Détail des profils")
        for c in clusters_data["clusters"]:
            with st.expander(f"{c['profil']} — {c['nb_arrets']} arrêts"):
                st.write(f"**Volume moyen :** {c['volume_moyen']:,.0f} validations")
                st.write(f"**Exemples :** {', '.join(c['exemples'])}")


# ============================================================
# PAGE 2 — PRÉDICTION COURT TERME (Random Forest)
# ============================================================
elif page == "Prédiction court terme":
    st.title("Prédiction du trafic — court terme")

    # Explication du fonctionnement, SANS figer de date.
    # La borne haute des données évolue au fil des ajouts : on la lit.
    st.markdown(
        '<div class="info-band">'
        'Ce modèle (Random Forest) estime le trafic d\'un arrêt à une date '
        'donnée. Il s\'appuie sur le trafic récent — notamment celui de la '
        'veille et du même jour la semaine précédente — combiné au jour de '
        'semaine, au mois, aux jours fériés et aux vacances.<br><br>'
        'Il est le plus fiable pour une date <b>proche des dernières données '
        'disponibles</b>. Plus la date s\'éloigne dans le futur, moins le '
        'trafic récent existe : le modèle s\'appuie alors davantage sur les '
        'tendances générales, et sa précision diminue. Pour une projection '
        'à plusieurs mois, préférez la page « Prédiction long terme ».'
        '</div>',
        unsafe_allow_html=True
    )

    if DATE_FIN:
        st.caption(f"Données disponibles jusqu'au {DATE_FIN}. "
                   f"À mesure que vous ajoutez des validations, cette limite "
                   f"avance et les prédictions proches gagnent en précision.")
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        arret = st.selectbox("Arrêt", arrets,
                             index=arrets.index("CHATELET") if "CHATELET" in arrets else 0)
        # Date par défaut : le lendemain de la dernière donnée, calculé
        # dynamiquement (pas de "1er juillet" en dur).
        defaut = (d_fin + timedelta(days=1)) if d_fin else date.today()
        date_pred = st.date_input("Date à prédire", value=defaut)

        # Repère visuel : la date demandée est-elle dans les données ou au-delà ?
        if d_fin and date_pred > d_fin:
            ecart = (date_pred - d_fin).days
            st.caption(f"Cette date est {ecart} jour(s) après la dernière "
                       f"donnée — c'est une vraie prédiction.")
        elif d_debut and d_fin and d_debut <= date_pred <= d_fin:
            st.caption("Cette date est dans les données connues — le modèle "
                       "restitue ce qu'il a appris (utile pour vérifier).")

        predire = st.button("Prédire le trafic", use_container_width=True)

    with col2:
        if predire:
            status, result = api_post("/predict/traffic",
                                      {"arret": arret, "date": str(date_pred)})
            if status == 200:
                st.markdown(f"### {result['predicted_validations']:,} validations")
                st.caption(f"Prédiction pour {arret} le {date_pred} "
                          f"(confiance : {result['confidence']:.0%})")

                if result.get("top_factors"):
                    st.markdown("**Facteurs les plus influents**")
                    df_fac = pd.DataFrame(result["top_factors"])
                    fig = go.Figure(go.Bar(
                        x=df_fac["shap"], y=df_fac["feature"],
                        orientation="h",
                        marker_color=["#2563a8" if s > 0 else "#d97a5a"
                                      for s in df_fac["shap"]]
                    ))
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                                      xaxis_title="Impact sur la prédiction")
                    st.plotly_chart(fig, use_container_width=True)
                    for f in result["top_factors"]:
                        st.caption(f"• {f['feature']} — {f['direction']}")
            elif status == 404:
                st.error(f"Arrêt introuvable : {result.get('detail')}")
            else:
                st.error(f"Erreur : {result.get('detail')}")
        else:
            st.info("Sélectionnez un arrêt et une date, puis lancez la prédiction.")

    st.divider()
    st.subheader("Contexte historique de l'arrêt")
    stats = api_get(f"/stats/{arret}")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trafic moyen", f"{stats['trafic_moyen']:,.0f}")
        c2.metric("Minimum", f"{stats['trafic_min']:,}")
        c3.metric("Maximum", f"{stats['trafic_max']:,}")
        c4.metric("Jours de données", stats['nb_jours'])


# ============================================================
# PAGE 3 — PRÉDICTION LONG TERME (Prophet)
# ============================================================
elif page == "Prédiction long terme":
    st.title("Prédiction du trafic — long terme")

    st.markdown(
        '<div class="info-band">'
        'Ce modèle (Prophet) prédit le trafic <b>plusieurs mois à l\'avance</b>. '
        'Contrairement au modèle court terme, il n\'a pas besoin du trafic '
        'récent : il apprend le rythme de l\'arrêt — sa tendance de fond, son '
        'cycle hebdomadaire, ses variations saisonnières, l\'effet des jours '
        'fériés — puis évalue n\'importe quelle date future.<br><br>'
        'La prédiction s\'accompagne d\'un <b>intervalle de confiance</b> : plus '
        'la date est lointaine, plus l\'incertitude grandit. À lire comme un '
        'ordre de grandeur, pas comme un chiffre exact.'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        arret = st.selectbox("Arrêt", arrets,
                             index=arrets.index("CHATELET") if "CHATELET" in arrets else 0)
        # Défaut : ~3 mois après la fin des données
        defaut = (d_fin + timedelta(days=90)) if d_fin else date.today() + timedelta(days=90)
        date_pred = st.date_input("Date à prédire", value=defaut)
        st.caption("Choisissez une date éloignée (plusieurs semaines ou mois "
                   "après les dernières données) pour exploiter ce modèle.")
        predire = st.button("Prédire", use_container_width=True)

    with col2:
        if predire:
            with st.spinner("Entraînement du modèle sur l'historique de "
                            "l'arrêt… (quelques secondes)"):
                status, result = api_post(
                    "/predict/long-term",
                    {"arret": arret, "date": str(date_pred), "avec_courbe": True}
                )

            if status == 200:
                pred = result["predicted_validations"]
                lo, hi = result["intervalle"]
                st.markdown(f"### {pred:,} validations")
                st.caption(f"Prédiction pour {arret} le {result['date']} "
                          f"({result['decomposition']['jour']}) — "
                          f"{result['horizon_jours']} jours après la dernière donnée")

                c1, c2, c3 = st.columns(3)
                c1.metric("Fourchette basse", f"{lo:,}")
                c2.metric("Estimation", f"{pred:,}")
                c3.metric("Fourchette haute", f"{hi:,}")
                st.caption(f"Intervalle de confiance à 80 % : la vraie valeur a "
                          f"80 % de chances de tomber entre {lo:,} et {hi:,}.")

                # ---- Graphique : historique + projection + bande ----
                if "courbe" in result:
                    hist = result["courbe"]["historique"]
                    proj = result["courbe"]["projection"]
                    fig = go.Figure()
                    # Bande d'incertitude
                    fig.add_trace(go.Scatter(
                        x=proj["dates"] + proj["dates"][::-1],
                        y=proj["borne_haute"] + proj["borne_basse"][::-1],
                        fill="toself", fillcolor="rgba(37,99,168,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                        name="Intervalle 80 %"
                    ))
                    # Historique réel
                    fig.add_trace(go.Scatter(
                        x=hist["dates"], y=hist["valeurs"],
                        mode="lines", line=dict(color="#1a2e44", width=1.3),
                        name="Historique"
                    ))
                    # Projection
                    fig.add_trace(go.Scatter(
                        x=proj["dates"], y=proj["valeurs"],
                        mode="lines", line=dict(color="#2563a8", width=1.8, dash="dash"),
                        name="Projection"
                    ))
                    # Point prédit
                    fig.add_trace(go.Scatter(
                        x=[result["date"]], y=[pred],
                        mode="markers", marker=dict(color="#d94a4a", size=10),
                        name="Date demandée"
                    ))
                    fig.update_layout(
                        height=380, margin=dict(l=0, r=0, t=20, b=0),
                        legend=dict(orientation="h", y=-0.15),
                        xaxis_title="", yaxis_title="Validations"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ---- Décomposition : pourquoi ce chiffre ? ----
                d = result["decomposition"]
                st.markdown("**Comment se compose cette prédiction**")
                st.caption(
                    f"Niveau de fond (tendance) : {d['tendance']:,} validations. "
                    f"Ajusté ensuite par le jour de semaine ({d['effet_semaine']:+.1f} %), "
                    f"la saison ({d['effet_saison']:+.1f} %) et les jours fériés "
                    f"({d['effet_ferie']:+.1f} %)."
                )
            elif status == 422:
                # Arrêt inéligible : trop peu d'historique ou trafic trop faible
                st.warning(result.get("detail", "Arrêt inéligible à la "
                                      "prédiction long terme."))
            else:
                st.error(f"Erreur : {result.get('detail')}")
        else:
            st.info("Sélectionnez un arrêt et une date lointaine, puis lancez "
                    "la prédiction. Le modèle s'entraîne à la volée — comptez "
                    "quelques secondes.")


# ============================================================
# PAGE 4 — DÉTECTION D'ANOMALIES
# ============================================================
elif page == "Détection d'anomalies":
    st.title("Détection d'anomalies")
    st.markdown("Vérifiez si un niveau de trafic est anormal pour un arrêt "
                "donné — utile pour repérer grèves, incidents ou événements.")
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        arret = st.selectbox("Arrêt", arrets,
                             index=arrets.index("GARE DU NORD") if "GARE DU NORD" in arrets else 0)

        stats = api_get(f"/stats/{arret}")
        if stats:
            st.caption(f"Trafic habituel : ~{stats['trafic_moyen']:,.0f} validations/jour")
            default_val = int(stats['trafic_moyen'])
        else:
            default_val = 5000

        nb_vald = st.number_input("Nombre de validations observé",
                                  min_value=0, value=default_val, step=100)
        detecter = st.button("Analyser", use_container_width=True)

    with col2:
        if detecter:
            status, result = api_post("/detect/anomaly",
                                      {"arret": arret, "nb_validations": nb_vald})
            if status == 200:
                if result["is_anomaly"]:
                    st.error(f"**Anomalie détectée**")
                else:
                    st.success(f"**Trafic normal**")
                st.markdown(result["message"])

                c1, c2 = st.columns(2)
                c1.metric("Z-score", f"{result['z_score']:.2f}")
                c2.metric("Moyenne de l'arrêt", f"{result['moyenne_arret']:,.0f}")

                z = result["z_score"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=nb_vald,
                    title={"text": "Validations observées"},
                    gauge={
                        "axis": {"range": [0, max(nb_vald, result['moyenne_arret']) * 1.5]},
                        "bar": {"color": "#d94a4a" if result["is_anomaly"] else "#2563a8"},
                        "threshold": {
                            "line": {"color": "gray", "width": 3},
                            "value": result['moyenne_arret']
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erreur : {result.get('detail')}")
        else:
            st.info("Sélectionnez un arrêt, saisissez un nombre de validations, puis analysez.")


# ============================================================
# PAGE 5 — PROFILS D'ARRÊTS (CLUSTERING)
# ============================================================
elif page == "Profils d'arrêts":
    st.title("Profils d'arrêts")
    st.markdown("Chaque arrêt appartient à une famille selon son comportement de trafic. "
                "Deux niveaux de détail sont disponibles.")
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        arret = st.selectbox("Arrêt", arrets,
                             index=arrets.index("VERSAILLES CH") if "VERSAILLES CH" in arrets else 0)
        version = st.radio("Niveau de détail",
                          ["Détaillé (5 profils)", "Simple (2 groupes)"])
        v = "k5" if "5" in version else "k2"
        analyser = st.button("Voir le profil", use_container_width=True)

    with col2:
        if analyser:
            result = api_get(f"/clusters/{arret}?version={v}")
            if result:
                st.markdown(f"### {result['profil']}")
                st.caption(f"{arret} — cluster {result['cluster']}")

                carac = result["caracteristiques"]
                c1, c2 = st.columns(2)
                c1.metric("Trafic semaine", f"{carac['trafic_moyen_semaine']:,.0f}")
                c2.metric("Trafic week-end", f"{carac['trafic_moyen_weekend']:,.0f}")
                c1.metric("Ratio week-end/semaine", f"{carac['ratio_weekend_semaine']:.2f}")
                c2.metric("Volume total", f"{carac['volume_total_92j']:,.0f}")

                st.markdown("**Arrêts au profil similaire**")
                for a in result["arrets_similaires"]:
                    st.write(f"• {a}")
            else:
                st.error("Arrêt introuvable ou modèle indisponible.")
        else:
            st.info("Sélectionnez un arrêt et un niveau de détail.")

    st.divider()
    st.subheader("Tous les profils")
    clusters_data = api_get(f"/clusters?version=k5")
    if clusters_data:
        df = pd.DataFrame(clusters_data["clusters"])
        fig = px.treemap(
            df, path=["profil"], values="nb_arrets",
            color="volume_moyen", color_continuous_scale="Blues"
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 6 — AJOUTER DES DONNÉES
# ============================================================
elif page == "Ajouter des données":
    st.title("Ajouter des données de validation")
    st.markdown("Enregistrez de nouvelles validations. Les données alimentent "
                "la base et permettront de réentraîner les modèles.")
    st.divider()

    CATEGORIES = ["Forfait Navigo", "Imagine R", "Amethyste", "Forfaits courts",
                  "Contrat Solidarité Transport", "Autres titres", "NON DEFINI"]

    mode = st.radio("Type d'arrêt", ["Arrêt existant", "Nouvel arrêt"],
                   horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        if mode == "Arrêt existant":
            arret = st.selectbox("Arrêt", arrets)
        else:
            arret = st.text_input("Nom du nouvel arrêt").strip().upper()
        jour = st.date_input("Date", value=date.today())
        categorie = st.selectbox("Catégorie de titre", CATEGORIES)
    with col2:
        nb_vald = st.number_input("Nombre de validations", min_value=0, value=1000, step=100)
        code_trns = st.text_input("Code transporteur", value="100")

    if st.button("Enregistrer", use_container_width=False):
        if not arret:
            st.warning("Veuillez saisir un nom d'arrêt.")
        else:
            payload = {
                "JOUR": str(jour),
                "LIBELLE_ARRET": arret,
                "CATEGORIE_TITRE": categorie,
                "NB_VALD": int(nb_vald),
                "CODE_STIF_TRNS": code_trns
            }
            status, result = api_post("/add-validations", payload)
            if status == 200:
                st.success(result["message"])
                st.info("Pour intégrer cette donnée aux modèles, régénérez le "
                       "Parquet (preprocessing.py) puis réentraînez.")
            else:
                st.error(f"Erreur : {result.get('detail')}")