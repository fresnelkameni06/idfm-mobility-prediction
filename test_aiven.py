# ============================================================
# Test de connexion à la base MySQL Aiven (cloud)
# Fichier : test_aiven.py (à la racine du projet)
# ============================================================
#
# POURQUOI CE SCRIPT ?
# Aiven impose une connexion SSL (ssl-mode=REQUIRED), contrairement
# au MySQL local. Ce script vérifie que la connexion chiffrée passe
# AVANT de créer la table et de brancher l'API. On isole le problème
# de connexion pour ne pas le découvrir plus tard, noyé dans le reste.
#
# PRÉ-REQUIS :
#   1. Un fichier .env.cloud à la racine (voir plus bas)
#   2. Le certificat CA d'Aiven dans data/external/ca.pem
#      (page du service Aiven → CA certificate → Show → télécharger)
#
# CONTENU DE .env.cloud :
#   DB_USER=avnadmin
#   DB_PASSWORD=votre_mot_de_passe_aiven
#   DB_HOST=mysql-idfm-fresnelkameni07-0724.l.aivencloud.com
#   DB_PORT=27252
#   DB_NAME=defaultdb
#   DB_SSL_CA=data/external/ca.pem
#
# LANCER :
#   python test_aiven.py
# ============================================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# On charge .env.cloud (et NON le .env local) pour viser Aiven
load_dotenv(".env.cloud")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_SSL_CA = os.getenv("DB_SSL_CA", "data/external/ca.pem")

# Vérification basique : les variables sont-elles bien présentes ?
manquantes = [k for k, v in {
    "DB_USER": DB_USER, "DB_PASSWORD": DB_PASSWORD, "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT, "DB_NAME": DB_NAME
}.items() if not v]
if manquantes:
    print("ERREUR — variables manquantes dans .env.cloud :", manquantes)
    print("Vérifiez que le fichier .env.cloud existe à la racine et est complet.")
    raise SystemExit(1)

if not os.path.exists(DB_SSL_CA):
    print(f"ERREUR — certificat CA introuvable : {DB_SSL_CA}")
    print("Téléchargez-le depuis Aiven (CA certificate → Show) et placez-le là.")
    raise SystemExit(1)

# L'URL de connexion. Le SSL est passé séparément via connect_args,
# car Aiven exige le certificat CA pour valider la connexion chiffrée.
url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(url, connect_args={"ssl": {"ca": DB_SSL_CA}})

try:
    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).fetchone()[0]
        print("Connexion Aiven OK — MySQL", version)

        tables = conn.execute(text("SHOW TABLES")).fetchall()
        noms = [t[0] for t in tables]
        print("Tables existantes :", noms if noms else "(aucune — base vide, normal)")
except Exception as e:
    print("ÉCHEC de la connexion :")
    print(" ", e)
    print("\nPistes :")
    print("  - Le mot de passe est-il correct dans .env.cloud ?")
    print("  - Le service Aiven est-il bien 'Running' ?")
    print("  - Le certificat ca.pem correspond-il à ce service ?")