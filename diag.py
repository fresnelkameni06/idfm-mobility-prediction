import pandas as pd, os
for label, p in [
    ("REFERENCE trafic/anomalies", "data/processed/idfm_features_reference.parquet"),
    ("COURANT   trafic/anomalies", "data/processed/idfm_features.parquet"),
    ("REFERENCE clustering",       "data/processed/arrets_clusters_reference.parquet"),
    ("COURANT   clustering",       "data/processed/arrets_clusters.parquet"),
]:
    if not os.path.exists(p):
        print(f"{label:28s} : ABSENT ({p})"); continue
    df = pd.read_parquet(p)
    info = f"{len(df):>7,} lignes"
    if 'JOUR' in df.columns:
        info += f" | {df['JOUR'].nunique():>3} jours | {df['JOUR'].min()} -> {df['JOUR'].max()}"
    if 'mois' in df.columns:
        info += f" | mois={sorted(df['mois'].unique())}"
    print(f"{label:28s} : {info}")