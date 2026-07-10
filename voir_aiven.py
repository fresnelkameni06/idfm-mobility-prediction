# ============================================================
# Voir le CONTENU de la table validations sur Aiven
# Fichier : voir_aiven.py
# ============================================================
# test_aiven.py liste les tables. Celui-ci montre les LIGNES :
# combien il y en a, et les dernières ajoutées. Utile pour vérifier
# que la page "Ajouter" écrit bien dans le cloud.
#   python voir_aiven.py
# ============================================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(".env.cloud")

url = (f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
       f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
engine = create_engine(url, connect_args={"ssl": {"ca": os.getenv("DB_SSL_CA")}})

with engine.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM validations")).fetchone()[0]
    print(f"Nombre de lignes dans Aiven : {total}")

    if total > 0:
        print("\nLes 10 dernières lignes ajoutées :")
        rows = conn.execute(text(
            "SELECT id, JOUR, LIBELLE_ARRET, CATEGORIE_TITRE, NB_VALD "
            "FROM validations ORDER BY id DESC LIMIT 10"
        )).fetchall()
        for r in rows:
            print(f"  #{r[0]}  {r[1]}  {r[2]}  {r[3]}  →  {r[4]} validations")                      
    else:
        print("\nLa table est VIDE. Aucun ajout n'est arrivé dans Aiven.")
        print("Pistes :") 
        print("  - Avez-vous relancé l'API après avoir remplacé predict.py ?")
        print("  - Le fichier .env.cloud est-il bien à la racine ?")
        print("  - L'ajout via l'interface a-t-il affiché un message de succès ?")   