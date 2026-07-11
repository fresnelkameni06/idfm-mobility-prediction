# Analyse & Prédiction des flux de mobilité — Île-de-France

Pipeline complet de *machine learning* appliqué aux validations de titres de transport du réseau francilien (Île-de-France Mobilités). Le projet couvre l'ensemble de la chaîne, de l'ingestion des données brutes jusqu'au déploiement en ligne : stockage SQL, nettoyage, *feature engineering*, entraînement de quatre modèles, exposition via une API REST, interface web interactive, conteneurisation Docker et intégration/déploiement continus.

**Démonstration en ligne** · Interface : `https://idfm-streamlit.onrender.com` — API : `https://idfm-api.onrender.com/docs`

> **À savoir avant de tester la démo en ligne** (contraintes des hébergements gratuits) :
>
> - **Réveil de Render (~50 s).** Les services Render gratuits se mettent en veille après quelques minutes d'inactivité. Au premier accès, la page peut mettre jusqu'à une minute à répondre, le temps que le service se réveille. Si l'interface semble figée ou affiche « API hors ligne », il suffit de **patienter puis de rafraîchir la page** : une fois réveillée, l'application répond normalement.
> - **Base Aiven en veille.** La base MySQL managée (offre gratuite) se met également en pause après une longue période sans usage. Cela n'affecte que la page « Ajouter des données » (qui écrit dans la base) ; les prédictions, elles, fonctionnent toujours car elles s'appuient sur le Parquet embarqué. Si un ajout échoue, c'est que la base se réveille : réessayer après quelques instants.
>
> Ces lenteurs sont propres aux offres gratuites et ne reflètent pas les performances réelles du projet, qui répond instantanément une fois les services actifs (ou en exécution locale).

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Les modèles](#les-modèles)
- [Structure du projet](#structure-du-projet)
- [Installation et exécution en local](#installation-et-exécution-en-local)
- [Déploiement en ligne](#déploiement-en-ligne)
- [L'API](#lapi)
- [Tests et intégration continue](#tests-et-intégration-continue)
- [Stack technique](#stack-technique)

---

## Aperçu

À partir des données ouvertes d'Île-de-France Mobilités (validations quotidiennes par arrêt et par catégorie de titre, sur une année glissante), le projet répond à quatre besoins métier :

1. **Prévoir le trafic à court terme** d'un arrêt pour une date proche, à partir de son historique récent.
2. **Prévoir le trafic à long terme**, plusieurs mois à l'avance, en apprenant les cycles hebdomadaires et saisonniers.
3. **Détecter les anomalies** de fréquentation (grèves, incidents, événements) par rapport au comportement habituel d'un arrêt.
4. **Segmenter les arrêts** en profils types selon leur usage (grands pôles, arrêts résidentiels, etc.).

Le jeu de données couvre environ **1,8 million de lignes**, **365 jours** et près de **780 arrêts** du réseau ferré.

---

## Architecture

Le projet sépare clairement les responsabilités : les données brutes vivent dans MySQL, les données enrichies dans un fichier Parquet que lisent les modèles, et l'API ne fait qu'exposer les prédictions sans jamais réentraîner.

```
                    ┌─────────────────┐
   CSV bruts  ───►  │  MySQL (local)  │  source de vérité, données brutes
   (IDFM)           └────────┬────────┘
                             │  preprocessing.py
                             ▼
                    ┌─────────────────┐
                    │  Parquet enrichi│  features calculées (lags, calendrier…)
                    └────────┬────────┘
                             │  train_*.py
                             ▼
                    ┌─────────────────┐
                    │  Modèles (.pkl) │  RandomForest, IsolationForest, K-Means
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
┌───────────────┐   ┌───────────────┐    ┌──────────────────┐
│  API FastAPI  │◄──│   Streamlit   │    │  MySQL cloud     │
│  (prédictions)│   │  (interface)  │───►│  (Aiven, ajouts) │
└───────────────┘   └───────────────┘    └──────────────────┘
```

**Principe clé :** les modèles lisent le Parquet, pas la base de données. La base MySQL managée (Aiven) ne sert qu'à collecter les nouvelles validations saisies via l'interface. Pour réentraîner, on rapatrie ces ajouts en local, on régénère le Parquet, puis on relance l'entraînement.

---

## Les modèles

| Modèle | Objectif | Algorithme | Sortie |
|--------|----------|------------|--------|
| **Court terme** | Trafic d'un arrêt à une date proche | Random Forest | Nombre de validations + confiance |
| **Long terme** | Trafic plusieurs mois à l'avance | Prophet | Prédiction + intervalle + décomposition |
| **Anomalies** | Fréquentation anormale | Isolation Forest + Z-score | Anomalie oui/non + score |
| **Clustering** | Profils d'arrêts | K-Means (k=2 et k=5) + PCA | Cluster + arrêts similaires |

**Court terme (Random Forest).** S'appuie sur des variables temporelles (jour de semaine, mois, jours fériés, vacances) et sur le trafic récent (trafic de la veille, du même jour la semaine précédente, moyennes glissantes). Idéal pour une date proche des dernières données disponibles.

**Long terme (Prophet).** Apprend une décomposition du temps : tendance de fond, cycle hebdomadaire, saisonnalité annuelle, effet des jours fériés. Contrairement au Random Forest, il n'a pas besoin du trafic récent et peut évaluer n'importe quelle date future. Il est entraîné à la volée sur l'historique de l'arrêt demandé. Deux garde-fous refusent les arrêts sans historique suffisant (moins de 180 jours) ou au trafic trop faible (moins de 500 validations/jour), où la saisonnalité ne serait pas fiable.

**Anomalies (Isolation Forest).** Combine un score Z (écart à la moyenne historique de l'arrêt) et un Isolation Forest pour signaler les jours de fréquentation inhabituelle.

**Clustering (K-Means).** Segmente les arrêts selon leur profil de trafic (ratio semaine/week-end, volume, variabilité) en deux niveaux de détail, avec visualisation par réduction de dimension PCA.

---

## Structure du projet

```
idfm-mobility/
├── api/
│   └── main.py                 API FastAPI — 9 endpoints
├── src/
│   ├── ingestion.py            CSV bruts → MySQL
│   ├── preprocessing.py        MySQL → Parquet enrichi (feature engineering)
│   ├── train_traffic.py        entraîne le modèle court terme (Random Forest)
│   ├── train_prophet.py        prépare le calendrier Prophet + validation
│   ├── train_anomaly.py        entraîne la détection d'anomalies
│   ├── train_clustering.py     entraîne le clustering des arrêts
│   └── predict.py              charge les modèles et sert les prédictions
├── streamlit_app/
│   └── app.py                  interface web (6 pages)
├── tests/
│   └── test_api.py             tests de l'API (pytest)
├── data/
│   ├── processed/              Parquet enrichi
│   └── external/               calendrier des jours fériés / vacances
├── models/                     modèles entraînés (.pkl, via Git LFS)
├── .github/workflows/
│   └── ci_cd.yml               pipeline CI/CD (tests + déploiement)
├── Dockerfile.api              image de l'API
├── Dockerfile.streamlit        image de l'interface
├── docker-compose.yml          orchestration locale des deux services
├── requirements.txt            dépendances complètes (développement)
└── requirements-docker.txt     dépendances allégées (runtime conteneur)
```

---

## Installation et exécution en local

### Prérequis

- Python 3.11 ou supérieur
- MySQL (pour l'ingestion et le réentraînement)
- Docker et Docker Compose (pour l'exécution conteneurisée)
- Git avec [Git LFS](https://git-lfs.github.com) (les modèles `.pkl` sont volumineux)

### 1. Récupérer le projet

```bash
git clone https://github.com/fresnelkameni06/idfm-mobility-prediction.git
cd idfm-mobility-prediction
git lfs pull            # télécharge les modèles versionnés via LFS
```

### 2. Environnement Python

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration

Créer un fichier `.env` à la racine pour la connexion MySQL locale :

```
DB_USER=root
DB_PASSWORD=votre_mot_de_passe
DB_HOST=localhost
DB_PORT=3306
DB_NAME=idfm_mobility
```

### 4. Reconstruire les données et les modèles (optionnel)

Les modèles entraînés sont déjà fournis (via Git LFS). Pour refaire le pipeline complet à partir des CSV bruts :

```bash
python src/ingestion.py         # CSV → MySQL
python src/preprocessing.py     # MySQL → Parquet enrichi
python src/train_traffic.py     # modèle court terme
python src/train_prophet.py     # calendrier Prophet
python src/train_anomaly.py     # détection d'anomalies
python src/train_clustering.py  # clustering
```

### 5. Lancer l'application

**Option A — les deux services séparément**

Dans un premier terminal, l'API :

```bash
uvicorn api.main:app --reload
```

Dans un second terminal, l'interface :

```bash
streamlit run streamlit_app/app.py
```

Streamlit affiche alors dans le terminal une adresse à ouvrir dans le navigateur :

```
Local URL: http://localhost:8501
```

Copier ce lien dans la barre d'adresse ouvre l'interface. La documentation de l'API est accessible sur `http://localhost:8000/docs`.

**Option B — tout en une commande avec Docker**

```bash
docker-compose up --build
```

Cette commande construit et lance les deux conteneurs (API + interface). Une fois le démarrage terminé, le terminal affiche l'adresse locale de l'interface :

```
Local URL: http://localhost:8501
```

Il suffit de copier ce lien et de le coller dans la barre d'adresse du navigateur pour ouvrir l'application. L'API, elle, est accessible sur `http://localhost:8000/docs`.

Pour tout arrêter : `Ctrl + C`, puis `docker-compose down`.

---

## Déploiement en ligne

Le projet est déployé sur **Render** (deux services conteneurisés : API et interface) et **Aiven** (base MySQL managée, gratuite, en SSL).

### Principe

- L'API et l'interface sont construites depuis leurs Dockerfile respectifs, directement à partir du dépôt GitHub.
- Les secrets (identifiants Aiven, certificat SSL) ne sont **jamais** dans le code : ils sont injectés via les variables d'environnement de Render et un *Secret File* pour le certificat.
- La base Aiven ne reçoit que les validations ajoutées via l'interface ; les prédictions s'appuient sur le Parquet embarqué dans l'image.

### Comportement des offres gratuites

Deux limites à connaître, liées aux paliers gratuits utilisés :

- **Mise en veille de Render.** Les services s'endorment après une période d'inactivité et se réveillent au premier accès (délai possible d'environ 50 secondes). Un rafraîchissement de la page suffit une fois le service actif. Un outil de *ping* périodique (par exemple UptimeRobot) peut maintenir le service éveillé si nécessaire.
- **Mise en veille d'Aiven.** La base MySQL managée se met en pause après une longue inactivité et se réveille à la connexion suivante. Seule la fonction d'ajout de données en dépend ; les prédictions restent disponibles en permanence.

### Variables d'environnement (côté API)

```
DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME   → connexion Aiven
DB_SSL_CA                                          → chemin du certificat SSL
```

### Interface

Un seul réglage : la variable `API_URL` pointe vers l'URL publique de l'API.

```
API_URL=https://idfm-api.onrender.com
```

### Mettre à jour la version en ligne

Il n'y a aucune commande Docker à lancer pour déployer : Render construit les images lui-même depuis GitHub. Mettre à jour la version en ligne revient simplement à pousser le code :

```bash
git add .
git commit -m "description de la modification"
git push
```

Le *push* déclenche le pipeline CI/CD : les tests s'exécutent, et si tout passe, Render redéploie automatiquement les deux services. En résumé :

- **En local** → `docker-compose up --build`, puis ouvrir le lien affiché.
- **En ligne** → `git push`, le reste est automatique.

---

## L'API

Neuf *endpoints*, documentés automatiquement via Swagger à l'adresse `/docs`.

| Méthode | Endpoint | Rôle |
|---------|----------|------|
| `GET` | `/health` | État de l'API et des modèles |
| `POST` | `/predict/traffic` | Prédiction court terme (Random Forest) |
| `POST` | `/predict/long-term` | Prédiction long terme (Prophet) |
| `POST` | `/detect/anomaly` | Détection d'anomalie |
| `GET` | `/clusters/{arret}` | Profil d'un arrêt |
| `GET` | `/clusters` | Liste des clusters |
| `GET` | `/arrets` | Liste des arrêts |
| `GET` | `/stats/{arret}` | Statistiques d'un arrêt |
| `POST` | `/add-validations` | Ajout d'une validation (→ Aiven) |

**Exemple — prédiction court terme :**

```bash
curl -X POST https://idfm-api.onrender.com/predict/traffic \
  -H "Content-Type: application/json" \
  -d '{"arret": "CHATELET", "date": "2026-07-01"}'
```

---

## Tests et intégration continue

Les tests (pytest) vérifient que l'API démarre et que ses *endpoints* répondent avec les bons codes HTTP (200 pour une requête valide, 404 pour un arrêt inconnu, 422 pour une entrée invalide).

```bash
pytest tests/ -v
```

**Pipeline CI/CD** (`.github/workflows/ci_cd.yml`) : à chaque *push* sur `main`, GitHub Actions exécute les tests, puis **ne déclenche le déploiement sur Render que si les tests réussissent**. Un code cassé ne peut donc pas atteindre la production.

```
push  →  tests  ──réussite──►  déploiement Render
                └──échec──►  arrêt, aucun déploiement
```

---

## Stack technique

**Données & ML** · Python, Pandas, NumPy, scikit-learn, Prophet, PyArrow (Parquet)
**Base de données** · MySQL, SQLAlchemy, PyMySQL, Aiven (managé, SSL)
**API & interface** · FastAPI, Uvicorn, Pydantic, Streamlit, Plotly
**Infrastructure** · Docker, Docker Compose, Render, Git LFS
**CI/CD** · GitHub Actions, pytest

---

*Projet personnel de science des données — chaîne complète de la donnée brute au déploiement.*