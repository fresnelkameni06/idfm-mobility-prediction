# ============================================================
# Création de la table 'validations' sur Aiven (cloud)
# Fichier : create_table_aiven.py
# ============================================================
#
# POURQUOI CE SCRIPT ?
# La base Aiven est vide. Avant que la page "Ajouter" puisse y écrire,
# il faut créer la table 'validations' avec la même structure que le
# MySQL local. On la crée vide : elle ne recevra que les ajouts faits
# via l'interface (les données historiques restent en local et servent
# à générer le Parquet, que l'API lit).
#
# LANCER :
#   python create_table_aiven.py
# ============================================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(".env.cloud")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_SSL_CA = os.getenv("DB_SSL_CA", "data/external/ca.pem")

url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(url, connect_args={"ssl": {"ca": DB_SSL_CA}})

# Structure identique au MySQL local. Les types correspondent aux
# colonnes que l'ingestion et la page "Ajouter" écrivent.
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS validations (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    JOUR            DATE          NOT NULL,
    CODE_STIF_TRNS  VARCHAR(10),
    CODE_STIF_RES   VARCHAR(10),
    CODE_STIF_ARRET VARCHAR(10),
    ID_ZDC          INT,
    LIBELLE_ARRET   VARCHAR(100)  NOT NULL,
    CATEGORIE_TITRE VARCHAR(50)   NOT NULL,
    NB_VALD         INT           NOT NULL,
    INDEX idx_jour (JOUR),
    INDEX idx_arret (LIBELLE_ARRET)
);
"""

with engine.connect() as conn:
    conn.execute(text(CREATE_SQL))
    conn.commit()
    print("Table 'validations' créée sur Aiven.")

    # Vérification
    cols = conn.execute(text("DESCRIBE validations")).fetchall()
    print(f"\n{len(cols)} colonnes :")
    for c in cols:
        print(f"  {c[0]:<18} {c[1]}")